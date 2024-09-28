const express = require('express');
const bodyParser = require('body-parser')

const app = express();

const port = 8080

let API_REGISTRIES = [];

exports.addAPIRegistry = (type, url, callback) => {
    API_REGISTRIES.push([type, url, callback])
}



app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static('pages'))

// Initializing Modules
const auth = require('./modules/auth')
const sensorReadings = require('./modules/sensor-readings')


API_REGISTRIES.forEach(element => {
    if (element[0] == 'get') {
        app.get(element[1], element[2]);
    } else {
        app.post(element[1], element[2]);
    }
});


app.get('/', (req,res) => {
    res.send('Err. Home page')
})

app.listen(port, () => {
    console.log("Listening on " + port)
})