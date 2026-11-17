import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("$ Data Analysis")

uploaded_file  = st.file_uploader(
    "",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df)

    st.write(len(df), "rows")
    st.write(len(df.columns), "columns")

    st.subheader("Filter Data")

    date_column = st.selectbox("Select the date column", df.columns)
    category_column = st.selectbox("Select the category column", df.columns)
    amount_column = st.selectbox("Select the amount column", df.columns)

    filtered_df = df.copy()

    filtered_df[date_column] = pd.to_datetime(
        filtered_df[date_column], errors   =   "coerce"
    )

    filtered_df = filtered_df.dropna(subset=[date_column])

    if not filtered_df.empty:
        start_date  =  filtered_df[date_column].min().date()
        end_date   =  filtered_df[date_column].max().date()

        selected_dates = st.date_input(
            "Select date range",
            value=(start_date, end_date)
        )

        if len(selected_dates)  ==  2:
            filtered_df = filtered_df[
                (filtered_df[date_column].dt.date >= selected_dates[0]) &
                (filtered_df[date_column].dt.date <= selected_dates[1])
            ]

    categories = df[category_column].dropna().unique()

    selected_category = st.selectbox(
        "Select category",
        categories
    )

    filtered_df = filtered_df[
        filtered_df[category_column] == selected_category
    ]
    filtered_df[amount_column] = pd.to_numeric(
        filtered_df[amount_column], errors="coerce"
    )

    filtered_df = filtered_df.dropna(subset=[amount_column])

    if not filtered_df.empty:
        min_amount = float(filtered_df[amount_column].min())   
        max_amount = float(filtered_df[amount_column].max())

        minimum = st.number_input(
            "smallest transaction",
            value=min_amount
        )

        maximum = st.number_input(
            "biggest transaction",
            value=max_amount
        )

        filtered_df = filtered_df[
            (filtered_df[amount_column] >= minimum) &
            (filtered_df[amount_column] <= maximum)
        ]

    st.subheader("New Data")

    st.dataframe(filtered_df)

    st.write(len(filtered_df), "transactions after filters")

    st.subheader("Overall stats")

    if not filtered_df.empty:

        amounts = filtered_df[amount_column]

        transaction_count = len(amounts)
        total_amount = amounts.sum()
        average_amount = amounts.mean()
        median_amount = amounts.median()
        minimum_amount = amounts.min()
        maximum_amount =  amounts.max()

        col1, col2, col3 = st.columns(3)

        col1.metric("Transactions", f"{transaction_count:,}")
        col2.metric("Total $", f"${total_amount:,.2f}")
        col3.metric("Average", f"${average_amount:,.2f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Median", f"${median_amount:,.2f}")
        col5.metric("Min", f"${minimum_amount:,.2f}")
        col6.metric("Max", f"${maximum_amount:,.2f}")

    # benford's first  digit
    st.subheader("Benford first digit")

    if not filtered_df.empty:

        # find $ amounts  and  remove  zeros
        benford_amounts = filtered_df[amount_column].abs()
        benford_amounts = benford_amounts[benford_amounts > 0]
 
        first_digits = benford_amounts.astype(str)
        first_digits = first_digits.str.replace(".", "", regex=False)
        first_digits = first_digits.str.lstrip("0")
        first_digits = first_digits.str[0].astype(int)

        actual_counts = first_digits.value_counts().reindex(
            range(1, 10), fill_value=0
        )

  
        actual_percent = actual_counts   /   len(first_digits) * 100


        digits  = np.arange(1, 10)
        expected_percent =  np.log10(1 + 1 / digits) * 100


        benford_df = pd.DataFrame({
            "Digit": digits,
            "Actual %": actual_percent.values,
            "Expected %": expected_percent
        })

        st.dataframe(benford_df.round(2), hide_index=True)

        fig, ax = plt.subplots()

        ax.bar(digits - 0.2, actual_percent.values, width=0.4, label="Actual")
        ax.bar(digits + 0.2, expected_percent, width=0.4, label="Expected")

        ax.set_xlabel("First Digit")
        ax.set_ylabel("Percentage")
        ax.set_xticks(digits)
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

        # benford's first  2
        st.subheader("Benford first two")

        two_digits = benford_amounts.astype(str)
        two_digits = two_digits.str.replace(".", "", regex=False)
        two_digits =  two_digits.str.lstrip("0")
        two_digits = two_digits.str[:2].astype(int)
        two_digits = two_digits[two_digits >= 10]

        if not two_digits.empty:

            digits_two = np.arange(10, 100)
            actual_two  =  two_digits.value_counts().reindex(
                range(10, 100), fill_value=0
            )/len(two_digits) * 100
            expected_two = np.log10(1 + 1 / digits_two) * 100   

            benford_two_df = pd.DataFrame({
                "Digits": digits_two,
                "Actual %": actual_two.values,
                "Expected %": expected_two
            })

            st.dataframe(benford_two_df.round(2), hide_index=True)
            fig2, ax2 = plt.subplots(figsize=(14, 5))

            ax2.bar(digits_two - 0.2, actual_two.values, width=0.4, label="Actual")
            ax2.bar(digits_two + 0.2, expected_two, width=0.4, label  = "Expected")

            ax2.set_xlabel("First Two Digits")
            ax2.set_ylabel("%")
            ax2.set_xticks(range(10, 100, 5))
            ax2.legend()

            st.pyplot(fig2)
            plt.close(fig2)
    st.subheader("Benford KS Results")
    test_results = []

    if not filtered_df.empty and not benford_amounts.empty:
        # method from class
        ks_first = np.max(np.abs(
            np.cumsum(actual_percent.values / 100) -
            np.cumsum(expected_percent / 100)
        ))
        limit_first = 1.36/np.sqrt(len(first_digits))
        result_first = "Different" if ks_first > limit_first else "No huge difference"   

        test_results.append({
            "Test": "First digit vs Benford",
            "KS Statistic": round(ks_first, 5),
            "5% Critical Value": round(limit_first, 5),
            "Result": result_first
        })
        if not two_digits.empty:
            ks_second = np.max(np.abs(
                np.cumsum(actual_two.values / 100) -
                np.cumsum(expected_two / 100)
            ))
            limit_second = 1.36/np.sqrt(len(two_digits))
            result_second = "Different" if ks_second > limit_second else "No huge difference"

            test_results.append({
                "Test": "First two digits vs Benford",
                "KS Statistic": round(ks_second, 5),
                "5% Critical Value": round(limit_second, 5),
                "Result": result_second
            })
    if test_results:
        st.dataframe(pd.DataFrame(test_results), hide_index=True)

    st.subheader("Compare Two Years")
    years = sorted(filtered_df[date_column].dt.year.dropna().unique())

    if len(years) >= 2:
        year_one = st.selectbox("First year", years, index=0)
        year_two = st.selectbox("Second year", years, index=1)

        if year_one == year_two:
            st.warning("Please select two different years.")
        else:
            amounts_one = np.sort(filtered_df.loc[
                filtered_df[date_column].dt.year == year_one, amount_column
            ].to_numpy())
            amounts_two = np.sort(filtered_df.loc[
                filtered_df[date_column].dt.year == year_two, amount_column
            ].to_numpy())

            all_values = np.unique(np.concatenate((amounts_one, amounts_two)))
            cumulative_one = np.searchsorted(amounts_one, all_values, side="right")  /   len(amounts_one)
            cumulative_two = np.searchsorted(amounts_two, all_values, side="right")/len(amounts_two)
            ks_years = np.max(np.abs(cumulative_one - cumulative_two))
            limit_years = 1.36 * np.sqrt(
                (len(amounts_one) + len(amounts_two)) / (len(amounts_one) * len(amounts_two))
            )
            result_years = "Different" if ks_years > limit_years else "No huge difference"
            st.write(f"{year_one}: {len(amounts_one):,} transactions")
            st.write(f"{year_two}: {len(amounts_two):,} transactions")
            st.write("KS Statistic:", round(ks_years, 5))
            st.write("5% Critical Value:", round(limit_years, 5))
            st.write("Conclusion:", result_years)
            st.caption("The KS test compares the distribution of transaction amounts. "
                        "5% cutoff is close.")

            test_results.append({
                "Test": f"{year_one} vs {year_two} amounts",
                "KS Statistic": round(ks_years, 5),
                "5% Critical Value": round(limit_years, 5),
                "Result": result_years
            })
    else:
        st.info("Need transactions from at least two years to compare.")

    st.subheader("Download Results")

    if not filtered_df.empty:
        from io import BytesIO

        summary = pd.DataFrame({
            "Statistic": ["Transactions", "Total", "Average", "Median", "Min", "Max"],
            "Value": [transaction_count, total_amount, average_amount,
                      median_amount, minimum_amount, maximum_amount]
        })

        st.download_button("Download Summary (CSV)",
                           summary.to_csv(index=False),
                           file_name="summary.csv")

        st.download_button("Download First-Digit Results (CSV)",
                           benford_df.to_csv(index=False),
                           file_name="digit.csv")

        if not two_digits.empty:
            st.download_button("Download First-Two-Digit Results (CSV)",
                               benford_two_df.to_csv(index=False),
                               file_name="2digits.csv")

        st.download_button("Download KS Results (CSV)",
                           pd.DataFrame(test_results).to_csv(index=False),
                           file_name="ks_results.csv")

        st.download_button("Download Filtered Data (CSV)",
                           filtered_df.to_csv(index=False).encode("utf-8"),
                           file_name="transactions.csv", mime="text/csv")

        first_chart  = BytesIO()
        fig.savefig(first_chart, format="png", bbox_inches="tight")
        st.download_button("Download First-Digit Chart (PNG)", first_chart.getvalue(), 
                           file_name="chart.png", mime="image/png")

        if not two_digits.empty:
            second_chart   =  BytesIO()
            fig2.savefig(second_chart, format="png", bbox_inches="tight") 
            st.download_button("Download First-Two-Digit Chart (PNG)", second_chart.getvalue(),
                               file_name="2chart.png", mime="image/png")