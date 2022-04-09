import pandas as pd
from copy import deepcopy as dcp

"""
#dobra klasa chain powinna się składać z residues, które składają się z atoms
class chain:
    def __init__(self, symbol: str = "A", len: int = 0):
        self.__symbol = symbol
        self.__len = len

class res:

class Contact:

    def __init__(self):
        self._obj1 =
        self._obj2 =

"""


class Contacts:
    # in from_file and from_table no protection from empty last col
    # it'd be nice to add distance method, out_tofile, out_totable
    def from_file(self, file: str):
        self.__df = pd.read_csv(file, sep="\t", lineterminator="\n")

    def from_lines(self, lines: list):
        l = [i.rstrip("\n").split() for i in lines]
        if len(l[0]) == 6:
            for i in l:
                i.append("0")
        self.__df = pd.DataFrame(l, columns=self.__cols)

    def from_table(self, table: list):
        self.__df = pd.DataFrame(table, columns=self.__cols)

    def __init__(self, *args):
        self.__cols = ["chain1", "nres1", "at1", "chain2", "nres2", "at2", "dist"]
        self.__i = -1
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg[0], str):
                self.from_lines(arg)
            elif isinstance(arg[0], list):
                self.from_table(arg)
            self.__df = self.__df.astype({"dist": float})
        else:
            print("Error: Invalid argument passed to Contacts")
        self._len = self.__df.shape[0]

    def mean_dist(self):
        return pd.DataFrame.mean(self.__df["dist"])

    def rmv_over_avg(self):
        self.__df.sort_values(["dist"])
        self.__df = self.__df[self.__df["dist"] < self.__df.mean_dist()]

    def rmv_over_median(self, coef: float):
        self.__df.sort_values(["dist"])
        no_to_rmv = coef * len(self.__df["dist"])
        self.__df = self.__df[:-int(no_to_rmv)]

    def rmv_over_cutoff(self, cutoff: float):
        self.__df = self.__df[self.__df["dist"] < cutoff]

    def __len__(self):
        return self.__df.shape[0]

    @property
    def get(self):
        return dcp(self.__df)

    def __call__(self):
        return dcp(self.__df)

    def __str__(self):
        return print(self.__df)

    def lower_over_two(self):
        dists = self.__df["dist"].values.tolist()
        for no in range(int(len(dists) / 2)):
            if dists[no * 2] > dists[no * 2 + 1]:
                dists[no * 2] = dists[no * 2 + 1]
        self.__df.assign(dist=dists)
        self.__df = self.__df.iloc[::2]

    def to_file(self, outf: str):
        with open(outf, 'w') as outfile:
            outfile.write(self.__df.to_string())

    def to_screen(self):
        pass

    def sort(self):
        self.__df = self.__df.sort_values([0, 1, 2, 3, 4, 5])

    def sort_dist(self, ascending: bool = False):
        self.__df.sort_values([6], ascending=ascending)

    def __add__(self, c2):
        c2.sort()
        # popraw by tylko nakładające się dodawały!
        self.__df[6] += c2[6]
        return self.__df

    def __iadd__(self, c2):
        c2.sort()
        # popraw by tylko nakładające się dodawały!
        self.__df[6] += c2[6]
        return self

    def __iter__(self):
        return self

    def __getitem__(self, i):
        return self.__df.loc[i].values.tolist()

    def __next__(self):
        if self.__i + 1 < len(self.__df):
            self.__i += 1
            return self.__df.loc[self.__i].values.tolist()
        else:
            self.__i = -1
            raise StopIteration()


if __name__ == "__main__":
    import sys

    dist_dat = "/home/matifortunka/Dokumenty/docking/specific_docking/HDock/allhits_both/0/618675bd08b81/output_dist.dat"

    if len(sys.argv) == 2:
        dist_dat = sys.argv[1]
    else:
        sys.exit("Wrong number of arguments given. Correct syntax: contacts.py output_dist.dat.")


    def find_lno_str(s: str, lines: list):
        for i, v in enumerate(lines):
            if s in v:
                return i
        return None


    dist_file = open(dist_dat, "r")
    dist_lines = dist_file.readlines()
    dist_file.close()
    mean_distances = []
    mean_names = []
    while 1:
        idx_start = find_lno_str("START", dist_lines)
        if idx_start is not None:
            name = dist_lines[idx_start].split()[1]
            idx_end = find_lno_str("END", dist_lines)
            mean_d = Contacts(dist_lines[idx_start + 1:idx_end])
            #mean_d.lower_over_two()
            mean_distances.append(mean_d.mean_dist())
            mean_names.append(name)
            dist_lines = dist_lines[idx_end + 1:]
        else:
            break

    all_means = pd.DataFrame(mean_distances, columns=["dist"])
    all_means.index = mean_names
    all_means = all_means.sort_values(by="dist")
    out = "mean_contacts.dat"
    with open(out, 'w') as outfile:
        outfile.write(all_means.to_string())
