const axios = require("axios");
const express = require("express");
const bodyParser = require("body-parser");
const { spawn } = require("child_process");
const path = require("path");
const cors = require("cors");

const app = express();
const port = 3000;

app.use(cors());
app.use(bodyParser.json());
app.use("/results", express.static(path.join(__dirname, "results")));
app.use("/crop_images", express.static(path.resolve(__dirname, "crop_images")));

app.post("/predict", (req, res) => {
  const { N, P, K, ph, location } = req.body;
  console.log(req.body);
  
  const lat = location?.latitude || 28.6139;  
  const lon = location?.longitude || 77.2090;

  const weatherAPI = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current_weather=true&hourly=relative_humidity_2m,precipitation&timezone=auto`;

  axios.get(weatherAPI)
    .then((response) => {
      const currentWeather = response.data.current_weather || {};
      const humidity = response.data?.hourly?.relative_humidity_2m?.[0] || 70;
      const rainfall = response.data?.hourly?.precipitation?.[0] || 3;

      const enrichedInput = {
        N,
        P,
        K,
        ph,
        temperature: currentWeather.temperature || 25,
        humidity,
        rainfall,
      };

      const userInput = JSON.stringify(enrichedInput);

      const scriptPath = path.resolve(__dirname, "predict_and_explain2.py");
      const pythonPath = "venv/bin/python3.10";  
      const py = spawn(pythonPath, [scriptPath, userInput]);

      console.log("spawn process done");

      let output = "";
      let error = "";

      py.stdout.on("data", (data) => {
        output += data.toString();
      });

      py.stderr.on("data", (data) => {
        error += data.toString();
      });

      py.on("close", (code) => {
        if (code !== 0) {
          console.error("❌ Python script failed with exit code:", code);
          console.error("Python stderr:", error);
          return res.status(500).send("Python execution failed:\n" + error);
        }

        try {
          const result = JSON.parse(output);
          return res.json(result);
        } catch (e) {
          console.error("❌ Failed to parse Python JSON:", e);
          console.error("Output was:", output);
          return res.status(500).send("Invalid JSON returned from Python.");
        }
      });
    })
    .catch((err) => {
      console.error("❌ Weather API call failed:", err);
      return res.status(500).send("Failed to fetch weather data.");
    });
});

app.listen(port, '0.0.0.0', () => {
  console.log(`✅ Crop Recommendation API running at http://localhost:${port}`);
});
