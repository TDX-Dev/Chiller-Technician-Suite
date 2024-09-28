
const inputs = document.getElementsByClassName("widget-input")

/** @type {HTMLCollectionOf<Element>} */
const pumpInfos = document.getElementsByClassName("pump-info-tile")

const myOnnxSession = new onnx.InferenceSession();

myOnnxSession.loadModel("./PumpFrequency.onnx");


function calculate() {
    let kiloWattTotal = parseFloat(inputs[0].value);
    let chillerLoad = parseFloat(inputs[1].value);
    let gpm = parseFloat(inputs[2].value);
    let cwst = parseFloat(inputs[3].value);
    let cwrt = parseFloat(inputs[4].value);
    let cdwst = parseFloat(inputs[5].value);
    let cdwrt = parseFloat(inputs[6].value);
    let wbt = parseFloat(inputs[7].value);
    let present_ch = parseFloat(inputs[8].value);
    let present_chp = parseFloat(inputs[9].value);
    let present_cds = parseFloat(inputs[10].value);
    let present_ct = parseFloat(inputs[11].value);



    let DeltaCHW = cwrt - cwst;
    let DeltaCDW = cdwrt - cdwst;
    let DeltaCT = parseFloat(inputs[12].value);

    for (let i = 0; i < inputs.length; i++) {
        if (inputs[i].value.trim() == "") {
            alert("Fill out all the fields");
            return;
        }
    }

    //const inputValues = [208.7, 39.6, 1238.0, 3.9, 47.0, 50.9, 3.9, 81.9, 78.0, 75.6, -2.4, 65.0, 11.2, 14.8, 9.1]
    const inputValues = [kiloWattTotal, chillerLoad, gpm, DeltaCHW, cwst, cwrt, DeltaCDW, cdwst, cdwrt, wbt, DeltaCT, present_ch, present_chp, present_cds, present_ct]

    const inputTensor = new onnx.Tensor(new Float32Array(inputValues), 'float32', [1, 15]);

    ['Hz_ CHP', 'Hz_CHS', 'Hz_CDS', 'Hz_CT']

    myOnnxSession.run([inputTensor]).then(
        (output) => {
            let outputTensor = output.values().next().value;
            pumpInfos[0].querySelector('h1').innerText = Math.round(outputTensor.data[0] * 100) / 100 + " Hz";
            pumpInfos[1].querySelector('h1').innerText = Math.round(outputTensor.data[1] * 100) / 100 + " Hz";
            pumpInfos[2].querySelector('h1').innerText = Math.round(outputTensor.data[2] * 100) / 100 + " Hz";

        }
    ).catch((err) => {
        console.error("Error during inference: ", err);
    });
    
}