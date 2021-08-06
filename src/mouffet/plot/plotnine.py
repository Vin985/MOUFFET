from plotnine import (
    aes,
    element_text,
    geom_line,
    ggplot,
    ggtitle,
    theme,
    theme_classic,
    save_as_pdf_pages,
)


def save_as_pdf(values, path):
    save_as_pdf_pages(values, path)


def plot_PR_curve(results, options):
    PR_df = results["stats"]

    plt = (
        ggplot(
            data=PR_df,
            mapping=aes(
                x=options.get("PR_curve_x", "recall"),
                y=options.get("PR_curve_y", "precision"),
            ),
        )
        + geom_line()
        + theme_classic()
        + theme(
            plot_title=element_text(weight="bold", size=14, margin={"t": 10, "b": 10}),
            figure_size=(20, 10),
            text=element_text(size=12, weight="bold"),
        )
        + ggtitle(
            (
                "Precision/Recall curve for model {}, database {}, class {}\n"
                + "with detector options {}"
            ).format(
                options["scenario_info"]["model"],
                options["scenario_info"]["database"],
                options["scenario_info"]["class"],
                options,
            )
        )
    )

    results["plots"].update({"PR_curve": plt})
    return results
