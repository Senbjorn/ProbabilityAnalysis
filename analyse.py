import sys
import numpy as np
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.Alphabet import IUPAC
from matplotlib import pyplot as plt

class AnalysisRes:
    def __init__(self, tensor, acid_dict):
        if not isinstance(tensor, np.ndarray):
            raise TypeError('parameter tensor must be an instance of numpy.ndarray.')
        self.tensor = tensor
        self.dim = len(self.tensor.shape)
        self.acid_dict = acid_dict

    def plot_heat(self):
        if self.dim == 1:
            self._plot_heat1d()
        elif self.dim == 2:
            self._plot_heat2d()
        elif self.dim == 3:
            self._plot_heat3d()
        else:
            raise TypeError("data must be not more then 3-dimensional.")

    def _plot_heat1d(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(len(self.acid_dict))
        labels = [""] * len(self.acid_dict)
        for key in self.acid_dict:
            labels[self.acid_dict[key]] = key
        ax.bar(x, self.tensor, tick_label=labels, color='g')
        ax.set_xlabel("amino acids", fontdict={'size':17})
        ax.set_ylabel("frequency", fontdict={'size':17})
        ax.set_title("1D result representation", fontdict={'size':20})
        plt.show()

    def _plot_heat2d(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(self.tensor, vmin=0, vmax=1, cmap='Blues', origin='lower')
        ax.set_xticks(np.arange(len(self.acid_dict)))
        ax.set_yticks(np.arange(len(self.acid_dict)))
        labels = [""] * len(self.acid_dict)
        for key in self.acid_dict:
            labels[self.acid_dict[key]] = key
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("condition amino acids", fontdict={'size':17})
        ax.set_ylabel("actual amino acids", fontdict={'size':17})
        ax.set_title("2D data representation", fontdict={'size':20})
        for i in range(len(self.acid_dict)):
            for j in range(len(self.acid_dict)):
                text = ax.text(j, i, "{0:.2f}".format(self.tensor[i, j]),
                               ha="center", va="center", color="y", fontdict={'size':6})
        plt.show()

    def _plot_heat3d(self):
        pass

def analyse(alignment, *args):
    acids = IUPAC.protein.letters + "U"
    acid_dict = {acids[i]: i for i in range(len(acids))}
    nacid = len(acids)
    if len(args) == 1:
        pos1 = args[0]
        total = 0
        vector = np.zeros(nacid)
        for record in alignment:
            total += 1
            if record.seq[pos1] in acid_dict:
                vector[acid_dict[record.seq[pos1]]] += 1;
        vector /= total
        return AnalysisRes(vector, acid_dict)
    elif len(args) == 2:
        pos1, pos2 = args
        total = np.zeros(nacid)
        matrix = np.zeros((nacid, nacid))
        for record in alignment:
            if record.seq[pos2] in acid_dict:
                ind2 = acid_dict[record.seq[pos2]]
                total[ind2] += 1
                if record.seq[pos1] in acid_dict:
                    matrix[acid_dict[record.seq[pos1]]][ind2] += 1
        for i in range(nacid):
            for j in range(nacid):
                if total[j] != 0:
                    matrix[i][j] /= total[j]
        return AnalysisRes(matrix, acid_dict)
    elif len(args) == 3:
        pos1, pos2, pos3 = args
        total = np.zeros((nacid, nacid))
        tensor = np.zeros((nacid, nacid, nacid))
        for record in alignment:
            if record.seq[pos2] in acid_dict and record.seq[pos3] in acid_dict:
                ind2, ind3 = acid_dict[record.seq[pos2]], acid_dict[record.seq[pos3]]
                total[ind2][ind3] += 1
                if record.seq[pos1] in acid_dict:
                    tensor[acid_dict[record.seq[pos1]]][ind2][ind3] += 1
        for i in range(nacid):
            for j in range(nacid):
                for k in range(nacid):
                    if total[j][k] != 0:
                        tensor[i][j][k] /= total[j][k]
        return AnalysisRes(tensor, acid_dict)
    else:
        raise TypeError('analyse() takes from 2 to 4 arguments but {0:d} were given.'.format(len(args) + 1))


if __name__ == "__main__":
    fin = sys.argv[1]
    print("Input file: {0}".format(fin))
    sequence_record = SeqIO.parse(fin, 'fasta')
    res = analyse(sequence_record, 160, 170)
    sequence_record = SeqIO.parse(fin, 'fasta')
    for rec in sequence_record:
        print(rec.seq)
    res.plot_heat()
    print("Done!")
