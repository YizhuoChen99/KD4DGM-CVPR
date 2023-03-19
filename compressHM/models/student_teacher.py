from __future__ import print_function
import torch
import torch.nn as nn
import torch.distributions as D

STD = 0.5


class StudentTeacher(nn.Module):
    def __init__(self, teacher_model, student_model, **kwargs):
        ''' Helper to keep the student-teacher architecture '''
        super(StudentTeacher, self).__init__()
        self.teacher = teacher_model
        self.student = student_model

        # grab the meta config and print for
        self.config = kwargs['kwargs']

    def wake_student(self, param):

        return self.student.wake(param)

    def sleep_student(self):

        return self.student.sleep()

    def distill(self):
        assert self.teacher is not None
        param_teacher = self.teacher.generate()
        param_student = self.student.generate_distill(param_teacher)

        pz1, pz2, pz3, pz4 = param_teacher['pz']
        muy1t, muy2t = param_teacher['muy']
        l1, l2, l3, l4 = param_student['logitz']
        muy1s, muy2s = param_student['muy']

        bcel = nn.BCEWithLogitsLoss(reduction='sum')
        loss = bcel(l1, pz1) + bcel(l2, pz2) + bcel(l3, pz3) + bcel(
            l4, pz4) + self.kl_gaussian(muy1t, muy1s) + self.kl_gaussian(
                muy2t, muy2s)

        loss = loss / self.config['batch_size']

        return loss

    def kl_gaussian(self, p_mu, q_mu):

        p = D.Normal(p_mu, STD)
        q = D.Normal(q_mu, STD)

        return torch.sum(D.kl_divergence(p, q))
