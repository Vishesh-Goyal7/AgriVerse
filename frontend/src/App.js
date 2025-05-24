import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const CONDITIONS = {
  N: "Nitrogen (kg/ha)",
  P: "Phosphorus (kg/ha)",
  K: "Potassium (kg/ha)",
  temperature: "Temperature (°C)",
  humidity: "Humidity (%)",
  ph: "pH",
  rainfall: "Rainfall (mm)"
};

function App() {
  const [conditions, setConditions] = useState([]);
  const [values, setValues] = useState({});
  const [results, setResults] = useState(null);

  const availableOptions = Object.keys(CONDITIONS).filter(
    (c) => !conditions.includes(c)
  );

  const handleAddCondition = () => {
    if (availableOptions.length > 0) {
      setConditions([...conditions, availableOptions[0]]);
    }
  };

  const handleChange = (index, field) => {
    const updated = [...conditions];
    updated[index] = field;
    setConditions(updated);
  };

  const handleValueChange = (field, val) => {
    setValues({ ...values, [field]: val });
  };

  const handleSubmit = async () => {
    const finalData = {};
    Object.keys(CONDITIONS).forEach((key) => {
      finalData[key] = conditions.includes(key)
        ? parseFloat(values[key]) || null
        : null;
    });

    try {
      const res = await axios.post("http://localhost:3001/predict", finalData);
      setResults(res.data);
    } catch (error) {
      alert("API call failed");
      console.error(error);
    }
  };

  return (
    <div className="App">
      <h1>Enter conditions :</h1>

      <table>
        <tbody>
          {conditions.map((cond, i) => (
            <tr key={i}>
              <td>
                <select
                  value={cond}
                  onChange={(e) => handleChange(i, e.target.value)}
                >
                  {Object.keys(CONDITIONS)
                    .filter((c) => !conditions.includes(c) || c === cond)
                    .map((opt) => (
                      <option key={opt} value={opt}>
                        {opt}
                      </option>
                    ))}
                </select>
              </td>
              <td>
                <input
                  type="number"
                  placeholder="Enter value"
                  value={values[cond] || ""}
                  onChange={(e) => handleValueChange(cond, e.target.value)}
                />
              </td>
              <td className="unit">{CONDITIONS[cond]}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <button onClick={handleAddCondition}>Add Condition</button>
      <button onClick={handleSubmit}>Submit</button>

      {results && (
        <div className="results-section">
          {results.top_predictions.map((pred, idx) => (
            <div key={idx} className="crop-result">
              <h2>{idx + 1}. {pred.crop.charAt(0).toUpperCase() + pred.crop.slice(1)} ({(pred.probability * 100).toFixed(2)}%)</h2>
              <img src={`http://localhost:3001/${pred.image_path}`} alt={pred.crop} className="crop-graph" />
              <div className="crop-report">
                <div className="crop-report-text">{pred.report}</div>
              </div>
            </div>
          ))}
        </div>
      )}

    </div>
  );
}

export default App;
