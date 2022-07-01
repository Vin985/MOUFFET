from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def save_as_pdf(values, path):
    pp = PdfPages(path)
    for x in values:
        pp.savefig(x.figure_)
        plt.close(fig=x.figure_)
    pp.close()
