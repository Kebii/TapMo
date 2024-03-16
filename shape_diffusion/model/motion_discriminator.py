import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.1, 0.1)
        m.bias.data.fill_(0.01)

class SelfAttention(nn.Module):
    def __init__(self, attention_size,
                 batch_first=False,
                 layers=1,
                 dropout=.0,
                 non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        modules = []
        for i in range(layers - 1):
            modules.append(nn.Linear(attention_size, attention_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))

        # last attention layer must output 1
        modules.append(nn.Linear(attention_size, 1))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))

        self.attention = nn.Sequential(*modules)
        self.attention.apply(init_weights) 
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, inputs):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.attention(inputs).squeeze()
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        # representations = weighted.sum(1).squeeze()
        representations = weighted.sum(1).squeeze()
        return representations, weighted, scores

class MotionDiscriminator(nn.Module):

    def __init__(self,
                 rnn_size=512,
                 input_size=300,
                 num_layers=2,
                 output_size=1,
                 feature_pool="concat",
                 attention_size=512,
                 attention_layers=1,
                 attention_dropout=0.5):

        super(MotionDiscriminator, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.feature_pool = feature_pool
        self.num_layers = num_layers
        self.attention_size = attention_size
        self.attention_layers = attention_layers
        self.attention_dropout = attention_dropout

        self.gru = nn.GRU(self.input_size, self.rnn_size, num_layers=num_layers)

        linear_size = self.rnn_size if not feature_pool == "concat" else self.rnn_size * 2
        # linear_size = self.rnn_size * 2

        if feature_pool == "attention" :
            self.attention = SelfAttention(attention_size=self.attention_size,
                                       layers=self.attention_layers,
                                       dropout=self.attention_dropout)
            self.char_linear1 = nn.Linear(256, rnn_size)
            self.relu = nn.ReLU()
            self.char_linear2 = nn.Linear(rnn_size, rnn_size)

        self.fc = nn.Linear(linear_size, output_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, sequence, char_feature=None):
        """
        sequence: of shape [batch_size, seq_len, input_size]
        """
        batchsize, seqlen, input_size = sequence.shape
        sequence = torch.transpose(sequence, 0, 1)
        outputs, state = self.gru(sequence)


        if self.feature_pool == "concat":
            outputs = F.relu(outputs)
            avg_pool = F.adaptive_avg_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            max_pool = F.adaptive_max_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            output = self.fc(torch.cat([avg_pool, max_pool], dim=1))
        elif self.feature_pool == "attention":
            outputs = outputs.permute(1, 0, 2)
            char_feat = self.char_linear1(char_feature.to(outputs.device))        # bs 512
            char_feat = self.char_linear2(self.relu(char_feat))
            outputs_char = torch.concat([outputs, char_feat.unsqueeze(1)], dim=1)
            y, feat, attentions = self.attention(outputs_char)
            output = self.fc(y)
        else:
            output = self.fc(outputs[-1])

        return self.sigmoid(output)
    


class MeshMotionDiscriminator(nn.Module):

    def __init__(self,
                 rnn_size=512,
                 input_size=54,
                 num_layers=2,
                 output_size=1,
                 feature_pool="concat",
                 attention_size=512,
                 attention_layers=1,
                 attention_dropout=0.5):

        super(MotionDiscriminator, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.feature_pool = feature_pool
        self.num_layers = num_layers
        self.attention_size = attention_size
        self.attention_layers = attention_layers
        self.attention_dropout = attention_dropout

        self.gru = nn.GRU(self.input_size, self.rnn_size, num_layers=num_layers)

        linear_size = self.rnn_size if not feature_pool == "concat" else self.rnn_size * 2
        # linear_size = self.rnn_size * 2

        if feature_pool == "attention" :
            self.attention = SelfAttention(attention_size=self.attention_size,
                                       layers=self.attention_layers,
                                       dropout=self.attention_dropout)
            self.char_linear1 = nn.Linear(256, rnn_size)
            self.relu = nn.ReLU()
            self.char_linear2 = nn.Linear(rnn_size, rnn_size)

        self.fc = nn.Linear(linear_size, output_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, sequence, char_feature=None):
        """
        sequence: of shape [batch_size, seq_len, input_size]
        """
        batchsize, seqlen, input_size = sequence.shape
        sequence = torch.transpose(sequence, 0, 1)
        outputs, state = self.gru(sequence)


        if self.feature_pool == "concat":
            outputs = F.relu(outputs)
            avg_pool = F.adaptive_avg_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            max_pool = F.adaptive_max_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            output = self.fc(torch.cat([avg_pool, max_pool], dim=1))
        elif self.feature_pool == "attention":
            outputs = outputs.permute(1, 0, 2)
            char_feat = self.char_linear1(char_feature.to(outputs.device))        # bs 512
            char_feat = self.char_linear2(self.relu(char_feat))
            outputs_char = torch.concat([outputs, char_feat.unsqueeze(1)], dim=1)
            y, feat, attentions = self.attention(outputs_char)
            output = self.fc(y)
        else:
            output = self.fc(outputs[-1])

        return self.sigmoid(output)
