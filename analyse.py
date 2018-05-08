import sys
import numpy as np
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.Alphabet import IUPAC


def analyse(alignment, *args):
    acids = IUPAC.protein.letters + "U"
    acid_dict = {acids[i]: i for i in range(len(acids))}
    nacid = len(acids)
    if len(args) == 1:
        total = 0
        pos1 = args[0]
        vector = [0] * nacid
        for record in alignment:
            total += 1
            if record.seq[pos1] in acid_dict:
                vector[acid_dict[record.seq[pos1]]] += 1;
        return np.array(vector) / total
    elif len(args) == 2:
        pos1, pos2 = args[0], args[1]
        total = [0] * nacid
        matrix = [[0] * nacid for i in range(nacid)]
        for record in alignment:
            if record.seq[pos1] in acid_dict and record.seq[pos2] in acid_dict:
                ind1, ind2 = acid_dict[record.seq[pos1]], acid_dict[record.seq[pos2]]
                total[ind2] += 1
                matrix[ind1][ind2] += 1
        matrix = np.array(matrix)
        for i in range(nacid):
            for j in range(nacid):
                if total[j] != 0:
                    matrix[i][j] /= total[j]
        return matrix
    elif len(args) == 3:
        pos1, pos2, pos3 = args[0], args[1], args[2]
        total = [[0] * nacid for i in range(nacid)]
        tensor = [[[0] * nacid for j in range(nacid)] for i in range(nacid)]
        for record in alignment:
            if record.seq[pos1] in acid_dict and record.seq[pos2] in acid_dict and record.seq[pos3] in acid_dict:
                ind1, ind2, ind3 = acid_dict[record.seq[pos1]], acid_dict[record.seq[pos2]], acid_dict[record.seq[pos3]]
                total[ind2][ind3] += 1
                tensor[ind1][ind2][ind3] += 1
        for i in range(nacid):
            for j in range(nacid):
                for k in range(nacid):
                    if total[j][k] != 0:
                        tensor[i][j][k] /= total[j][k]
    else:
        raise TypeError('analyse() takes from 2 to 4 arguments but {0:d} were given.'.format(len(args) + 1))


if __name__ == "__main__":
    fin = sys.argv[1]
    print("Input file: {0}".format(fin))
    sequence_record = SeqIO.parse(fin, 'fasta')
    print(*analyse(sequence_record, 3, 8), sep="\n")
    print("Done!")
