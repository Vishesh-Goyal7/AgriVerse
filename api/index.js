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
  const userInput = JSON.stringify(req.body);
  console.log(userInput);
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
});

app.listen(port, '0.0.0.0', () => {
  console.log(`✅ Crop Recommendation API running at http://localhost:${port}`);
});
