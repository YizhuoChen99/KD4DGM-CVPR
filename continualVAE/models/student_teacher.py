from __future__ import print_function
import torch
import torch.nn as nn

from helpers.distributions import nll_activation, kl_out, WS22_gaussian


def detach_from_graph(param_map):
    for _, v in param_map.items():
        if isinstance(v, dict):
            detach_from_graph(v)
        else:
            v = v.detach_()


class StudentTeacher(nn.Module):
    def __init__(self, teacher_model, student_model, **kwargs):
        ''' Helper to keep the student-teacher architecture '''
        super(StudentTeacher, self).__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.continual_step = 0

        # grab the meta config and print for
        self.config = kwargs['kwargs']

    def loss_function(self, output_map, beta, alpha):
        result = {}

        x = output_map['x']
        x_reconstr_logits_student = output_map['student']['x_reconstr_logits']
        params_student = output_map['student']['params']

        elbo, elbo_nll, kl1, kl2, kl3, kl4, kl5 = self.student.nelbo(
            x, x_reconstr_logits_student, params_student, beta)

        loss = elbo

        result.update({
            'loss': loss,
            'loss_mean': torch.mean(loss),
            'elbo_mean': torch.mean(elbo),
            'elbo_nll_mean': torch.mean(elbo_nll),
            'kl1_mean': torch.mean(kl1),
            'kl2_mean': torch.mean(kl2),
            'kl3_mean': torch.mean(kl3),
            'kl4_mean': torch.mean(kl4),
            'kl5_mean': torch.mean(kl5)
        })

        if 'distill' in output_map:
            gen_logits_teacher = output_map['distill']['gen_logits_teacher']
            gen_logits_student = output_map['distill']['gen_logits_student']
            params_gen_teacher = output_map['distill']['params_gen_teacher']
            params_gen_student = output_map['distill']['params_gen_student']

            dpkl = []
            for layer_index in (0, 1, 2, 3):

                diss_p = WS22_gaussian(
                    *params_gen_teacher['p'][layer_index],
                    *params_gen_student['p'][layer_index],
                    layer_reduction=self.config['distill_z_reduction'])

                dpkl.append(diss_p)

            doutkl = kl_out(gen_logits_student, gen_logits_teacher,
                            self.config['nll_type'])

            dzkl = sum(dpkl)

            distill_loss = (doutkl +
                            self.config['distill_z_kl_lambda'] * dzkl) * alpha
            loss = torch.cat(
                (distill_loss *
                 (1 - self.config['ratio']), elbo * self.config['ratio']),
                dim=0)

            result.update({
                'loss': loss,
                'loss_mean': torch.mean(loss),
                'distill_mean': torch.mean(distill_loss),
                'dpkl1_mean': torch.mean(dpkl[0]),
                'dpkl2_mean': torch.mean(dpkl[1]),
                'dpkl3_mean': torch.mean(dpkl[2]),
                'dpkl4_mean': torch.mean(dpkl[3]),
                'doutkl_mean': torch.mean(doutkl)
            })

        return result

    def forward(self, x):
        condition = self.student.generate_condition(x.shape[0], 0)
        x_reconstr_logits, params_student = self.student(x, condition)
        x_reconstr = nll_activation(x_reconstr_logits, self.config['nll_type'])

        ret_map = {
            'student': {
                'params': params_student,
                'x_reconstr': x_reconstr,
                'x_reconstr_logits': x_reconstr_logits
            },
            'x': x,
        }

        return ret_map

    def distill_forward(self, x):
        condition = self.student.generate_condition(x.shape[0], 1)
        x_reconstr_logits, params_student = self.student(x, condition)
        x_reconstr = nll_activation(x_reconstr_logits, self.config['nll_type'])

        ret_map = {
            'student': {
                'params': params_student,
                'x_reconstr': x_reconstr,
                'x_reconstr_logits': x_reconstr_logits
            },
            'x': x,
        }

        gen_size = self.config['batch_size']
        gen_logits_teacher, gen_teacher, params_gen_teacher = self.teacher.generate_synthetic_samples(
            gen_size, 0)

        gen_logits_student, gen_student, params_gen_student = self.student.generate_synthetic_samples(
            gen_size, 0, noise_list=params_gen_teacher['noise'])

        ret_map.update({
            'distill': {
                'gen_logits_teacher': gen_logits_teacher,
                'gen_logits_student': gen_logits_student,
                'gen_teacher': gen_teacher,
                'gen_student': gen_student,
                'params_gen_teacher': params_gen_teacher,
                'params_gen_student': params_gen_student
            },
        })
        return ret_map
