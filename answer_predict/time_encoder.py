import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from datetime import date

def to_variable(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad = requires_grad)

class TimeEncoder:
    """Implement the Time Encoding Function"""
    def __init__(self, num_dim, dropout, span, min_date, max_date):
        super(TimeEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.decay_rate = 1
        self.span = span
        self.min_date = min_date
        self.max_date = max_date
        print ("min_date:", self.min_date)
        print("max_date:", self.max_date)
        self.date_margin = 10
        max_time_span = self.time_span(self.min_date, self.max_date) + 1 + self.date_margin
        print ("max_time_span", max_time_span)
        # Compute the time encodings once in log space.
        self.time_encode = torch.zeros(max_time_span, num_dim)
        position = torch.arange(0., max_time_span).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., num_dim, 2) *
                             -(math.log(10000.0) / num_dim))
        self.time_encode[:, 0::2] = torch.sin(position * div_term)
        self.time_encode[:, 1::2] = torch.cos(position * div_term)

    def time_span(self, min_date, max_date):
        print (min_date["year"])
        X = min_date["year"] - 1
        max_date["year"] = max_date["year"] - X
        from_date = date(1, min_date["month"], min_date["day"])
        to_date = date(max_date["year"], max_date["month"], max_date["day"])
        delta = to_date - from_date
        idx = int(delta.days) + self.date_margin
        return idx

    def temporal_index(self, encode_date):
        X = self.min_date["year"] - 1
        encode_date["year"] = encode_date["year"] - X
        from_date = date(1, self.min_date["month"], self.min_date["day"])
        try:
            to_date = date(encode_date["year"], encode_date["month"], encode_date["day"])
            delta = to_date - from_date
            idx = int(delta.days) + self.date_margin
            return idx
        except ValueError:
            return None

    def get_time_encoding(self, date, num_shift=0):
        idx = self.temporal_index(date)
        if idx is not None:
            time_encode = self.time_encode[idx, :]
            for i in range(1, num_shift + 1):
                time_encode_i_pre = self.time_encode[idx - i, :]
                time_encode_i_post = self.time_encode[idx + i, :]
                time_encode += (time_encode_i_pre + time_encode_i_post) * (self.decay_rate ** i)
            time_encode /= (num_shift * 2 + 1)
            return to_variable(time_encode, requires_grad=False)
        else:
            return None