function averageMonthlyData(data) {
    const monthlyData = {};

    // Step 1: Loop through the data
    data.forEach(item => {
        const date = new Date(item.Time); // Using the 'Time' field as the timestamp
        const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;

        // Initialize the month data if not present
        if (!monthlyData[monthKey]) {
            monthlyData[monthKey] = { sum: {}, count: 0 };
        }

        // Sum up the values for each column except 'Time'
        for (let key in item) {
            if (key !== 'Time') {
                if (!monthlyData[monthKey].sum[key]) {
                    monthlyData[monthKey].sum[key] = 0;
                }
                monthlyData[monthKey].sum[key] += parseFloat(item[key]) || 0;
            }
        }

        // Increment the count for the month
        monthlyData[monthKey].count += 1;
    });

    // Step 2: Compute the average for each month
    const monthlyAverages = [];

    for (let month = 0; month < 12; month++) {
        const year = new Date().getFullYear(); // Assuming data is from the current year
        const monthKey = `${year}-${String(month + 1).padStart(2, '0')}`;

        if (monthlyData[monthKey]) {
            const { sum, count } = monthlyData[monthKey];
            const avg = {};

            for (let key in sum) {
                avg[key] = sum[key] / count;
            }

            avg['Time'] = monthKey; // Assign the month key to 'Time' for reference
            monthlyAverages.push(avg);
        } else {
            // If there is no data for the month, fill with zeros
            const zeroEntry = { Time: monthKey };
            for (let key in data[0]) {
                if (key !== 'Time') {
                    zeroEntry[key] = 0;
                }
            }
            monthlyAverages.push(zeroEntry);
        }
    }

    return monthlyAverages;
}
