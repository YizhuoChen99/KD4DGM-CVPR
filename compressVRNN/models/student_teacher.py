from __future__ import print_function
import torch
import torch.nn as nn
import torch.distributions as D


class StudentTeacher(nn.Module):
    def __init__(self, teacher_model, student_model, **kwargs):
        ''' Helper to keep the student-teacher architecture '''
        super(StudentTeacher, self).__init__()
        self.teacher = teacher_model
        self.student = student_model

        # grab the meta config and print for
        self.config = kwargs['kwargs']

    def forward_student(self, x, lengths):
        return self.student.forward(x, lengths)

    def distill(self):
        assert self.teacher is not None
        param_teacher = self.teacher.generate(1.0, 0, 1.0)
        return self.student.generate_distill(param_teacher)
