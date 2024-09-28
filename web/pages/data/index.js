const canvas_pwr_consum_ctx = addCanvas('chiller_pwr_consum', 'contId', 'Chiller Power Consumption');

const canvas_efficiency_ctx = addCanvas('chiller_efficiency', 'contId', 'Chiller Efficiency');

const canvas_chiller_load = addCanvas('chiller_load', 'contId', 'Chiller Load (%)')

const canvas_chiller_main_pump = addCanvas('chiller_primary_pump', 'contId', 'Chiller Primary Pump (Hz)')

const myOnnxSession = new onnx.InferenceSession();

myOnnxSession.loadModel('./PumpFrequency.onnx')

// CREATING CHARTS

function addChart(ctx, labels, datasets) {
    return new Chart(
        ctx,
        {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            }


        }
    )
}

let chiller_pwr_consum_chart = addChart(
    canvas_pwr_consum_ctx,
    MONTHS,
    [
        {
            label: 'Previous Year',
            data: [60, 44, 113, 235, 64, 113, 275, 64, 234, 454, 567, 345],
            fill: false,
            tension: 0.1
        },
        {
            label: 'Predicted Year',
            data: [60, 128, 113, 275, 64, 234, 454, 567, 345, 234, 577, 989],
            fill: false,
            tension: 0.1
        }
    ]
)

let chiller_pwr_efficiency_chart = addChart(
    canvas_efficiency_ctx,
    MONTHS,
    [
        {
            label: 'Previous Year',
            data: [60, 450, 193, 235, 64, 700, 455, 294, 334, 245, 575, 456],
            fill: false,
            tension: 0.1
        },
        {
            label: 'Predicted Year',
            data: [60, 128, 700, 455, 294, 334, 245, 575, 123, 234, 565, 989],
            fill: false,
            tension: 0.1
        }
    ]
)

let chiller_chiller_load_chart = addChart(
    canvas_chiller_load,
    MONTHS,
    [
        {
            label: 'Previous Year',
            data: [60, 450, 193, 235, 64, 700, 455, 294, 334, 245, 575, 456],
            fill: false,
            tension: 0.1
        },
        {
            label: 'Predicted Year',
            data: [60, 128, 700, 455, 294, 334, 245, 575, 123, 234, 565, 989],
            fill: false,
            tension: 0.1
        }
    ]
)

let chiller_main_pump_chart = addChart(
    canvas_chiller_main_pump,
    MONTHS,
    [
        {
            label: 'Previous Year',
            data: [60, 450, 193, 235, 64, 700, 455, 294, 334, 245, 575, 456],
            fill: false,
            tension: 0.1
        },
        {
            label: 'Predicted Year',
            data: [60, 128, 700, 455, 294, 334, 245, 575, 123, 234, 565, 989],
            fill: false,
            tension: 0.1
        }
    ]
)

function getTimeOfDay(hour) {
    if (hour >= 0 && hour < 5) {
        return 3; // Night
    } else if (hour >= 5 && hour < 11) {
        return 0; // Morning
    } else if (hour >= 11 && hour < 17) {
        return 1; // Afternoon
    } else if (hour >= 17 && hour < 24) {
        return 2; // Evening
    } else {
        throw new Error('Hour must be between 0 and 23');
    }
}

let monthSelector = document.getElementById("month-selector");

monthSelector.addEventListener('input', function () {
    // Callback when value changes (e.g., through arrows or manual input)
    const daySelector = document.getElementById("day-selector");
    daySelector.setAttribute("max", daysInMonth[parseInt(monthSelector.value)])
    daySelector.value = "0";
});

const daysInMonth = {
    1: 31,  // January
    2: 28,  // February (non-leap year)
    3: 31,  // March
    4: 30,  // April
    5: 31,  // May
    6: 30,  // June
    7: 31,  // July
    8: 31,  // August
    9: 30,  // September
    10: 31, // October
    11: 30, // November
    12: 31  // December
};

const seasonMap = {
    12: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 1,
    5: 1,
    6: 2,
    7: 2,
    8: 2,
    9: 3,
    10: 3,
    11: 3
}

function calculateChillerLoad() {

    let inputs = document.getElementsByClassName("number-input-widget")

    let hour = parseFloat(inputs[2].value);
    let timeOfDay = getTimeOfDay(hour)
    let dayOfWeek = parseFloat(inputs[1].value);
    let season = seasonMap[parseFloat(inputs[0].value)]
    let month = parseFloat(inputs[0].value);
    let is_weekend = (dayOfWeek >= 5 ? 1 : 0)

    let textInputs = document.getElementsByClassName('widget-input')

    let rt = parseFloat(textInputs[0].value);
    let kW_Tot = parseFloat(textInputs[1].value);
    let kW_CHH = parseFloat(textInputs[2].value);
    let DeltaCHW = parseFloat(textInputs[3].value);
    let DeltaCDW = parseFloat(textInputs[4].value);

    for (let i = 0; i < inputs.length; i++) {
        if (inputs[i].value.trim() == "") {
            alert("Fill out all the fields.")
            return;
        }
    }
    for (let i = 0; i < textInputs.length; i++) {
        if (textInputs[i].value.trim() == "") {
            alert("Fill out all the fields.")
            return;
        }
    }

    console.log(inputs[0])
    const inputValues = [month, season, dayOfWeek, timeOfDay, rt, kW_Tot, kW_CHH, DeltaCHW, DeltaCDW]


    const inputTensor = new onnx.Tensor(new Float32Array(inputValues), 'float32', [1, 9]);

    console.log(new Float32Array(inputValues))


    myOnnxSession.run([inputTensor]).then(
        (output) => {


            let outputTensor = output.values().next().value;

            document.getElementById('chiller-usage').innerText = Math.abs(Math.round(outputTensor.data[0] * 100) / 100) + "%";
        }
    )
}

async function uploadData() {
    var input = document.createElement('input');
    input.type = 'file';
    input.click();

    input.onchange = e => {
        // getting a hold of the file reference
        var file = e.target.files[0];

        // setting up the reader
        var reader = new FileReader();
        reader.readAsText(file, 'UTF-8');

        // here we tell the reader what to do when it's done reading...
        reader.onload = readerEvent => {
            var content = readerEvent.target.result; // this is the content!
            const objects = $.csv.toObjects(content)

            const avgData = averageMonthlyData(objects);

            console.log(avgData);

            chiller_pwr_consum_chart.data.datasets[0] = {
                label: 'Previous Year',
                data: avgData.map(e=> {
                    return e.kW_Tot
                }),
                fill: false,
                tension: 0.1,
                borderColor: "#1E9EF5"
            }
            chiller_pwr_consum_chart.update()

            chiller_pwr_efficiency_chart.data.datasets[0] = {
                label: 'Previous Year',
                data: avgData.map(e=> {
                    return e.kW_RT
                }),
                fill: false,
                tension: 0.1,
                borderColor: "#1E9EF5"
            }
            chiller_pwr_efficiency_chart.update()

            chiller_chiller_load_chart.data.datasets[0] = {
                label: 'Previous Year',
                data: avgData.map(e=> {
                    return e["CH Load"]
                }),
                fill: false,
                tension: 0.1,
                borderColor: "#1E9EF5"
            }
            chiller_chiller_load_chart.update()

            chiller_main_pump_chart.data.datasets[0] = {
                label: 'Previous Year',
                data: avgData.map(e=> {
                    return e["Hz_CDS"]
                }),
                fill: false,
                tension: 0.1,
                borderColor: "#1E9EF5"
            }
            chiller_main_pump_chart.update()
            
        }
    }
}