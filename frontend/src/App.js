import React, { useState, useEffect, useRef } from "react";
import Chart from "chart.js/auto";
import "./App.css";

const allFeatures = [
  "N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"
];

function App() {
  const [rows, setRows] = useState([{ feature: "", value: "" }]);
  const [usedFeatures, setUsedFeatures] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [responseData, setResponseData] = useState(null);
  const [activeCrop, setActiveCrop] = useState(null);

  const chartRef = useRef(null);
  const chartInstanceRef = useRef(null);

  const handleAddRow = () => {
    setRows([...rows, { feature: "", value: "" }]);
  };

  const handleFeatureChange = (index, value) => {
    const updated = [...rows];
    updated[index].feature = value;
    setRows(updated);
    setUsedFeatures(updated.map(r => r.feature).filter(f => f));
  };

  const handleValueChange = (index, value) => {
    const updated = [...rows];
    updated[index].value = value;
    setRows(updated);
  };

  const handleSubmit = async () => {
    const inputObj = {};
    rows.forEach(row => {
      if (row.feature) inputObj[row.feature] = row.value ? parseFloat(row.value) : null;
    });

    try {
      const response = await fetch("http://localhost:3001/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputObj),
      });

      const data = await response.json();
      setPredictions(data.top_predictions || []);
      setResponseData(data);
    } catch (err) {
      alert("API call failed.");
      console.error(err);
    }
  };

  useEffect(() => {
    if (activeCrop && chartRef.current) {
      const ctx = chartRef.current.getContext("2d");

      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
      }

      const labels = activeCrop.feature_impact.map(f => f.feature);
      const values = activeCrop.feature_impact.map(f => f.shap);

      chartInstanceRef.current = new Chart(ctx, {
        type: "bar",
        data: {
          labels,
          datasets: [{
            label: "SHAP Contribution",
            data: values,
            backgroundColor: values.map(v => v >= 0 ? "#4caf50" : "#e53935")
          }]
        },
        options: {
          responsive: true,
          indexAxis: "y",
          plugins: {
            legend: { display: false },
            title: {
              display: true,
              text: "Feature Impact (SHAP Values)",
              font: { size: 18, family: "Georgia" }
            }
          },
          scales: {
            x: { title: { display: true, text: "SHAP Value" } },
            y: { ticks: { font: { family: "Georgia" } } }
          }
        }
      });
    }
  }, [activeCrop]);

  return (
    <div className="app-container">
      <header className="header">Crop Recommendation System</header>

      <div className="form-container">
        <table className="input-table">
          <thead>
            <tr>
              <th>Feature</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, idx) => (
              <tr key={idx}>
                <td>
                  <select
                    value={row.feature}
                    onChange={(e) => handleFeatureChange(idx, e.target.value)}
                  >
                    <option value="">Select</option>
                    {allFeatures
                      .filter(f => !usedFeatures.includes(f) || f === row.feature)
                      .map(f => (
                        <option key={f} value={f}>{f}</option>
                      ))}
                  </select>
                </td>
                <td>
                  <input
                    type="number"
                    value={row.value}
                    onChange={(e) => handleValueChange(idx, e.target.value)}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="button-group">
          <button onClick={handleAddRow}>Add Condition</button>
          <button onClick={handleSubmit}>Submit</button>
        </div>
      </div>

      <div className="results-section">
        {predictions.map(pred => (
          <div key={pred.crop} className="crop-result">
            <img
              src={`http://localhost:3001/crop_images/${pred.crop.toLowerCase()}.jpeg`}
              alt={pred.crop}
              className="crop-image"
              onClick={() => setActiveCrop(pred)}
            />
            <div className="crop-label">{pred.crop.charAt(0).toUpperCase() + pred.crop.slice(1)}</div>
          </div>
        ))}
      </div>

      {predictions.length > 0 && responseData?.trust_score && (
        <div className="trust-score">
          <h3>🔒 Trust Score</h3>
          <p><strong>Level:</strong> {responseData.trust_score.level}</p>
          <p><strong>Confidence Margin:</strong> {(responseData.trust_score.confidence * 100).toFixed(2)}%</p>
        </div>
      )}

      {console.log(responseData)}
      {responseData?.trust_score.counterfactual_suggestion && (
        <div className="counterfactual">
          <h3>♻️ Alternative Crop Suggestion</h3>
          <p>
            <strong>{responseData.trust_score.counterfactual_suggestion.alternative_crop}</strong> can also be grown
            with a minimal deviation of <strong>{responseData.trust_score.counterfactual_suggestion.percent_deviation}%</strong>.
          </p>
          <p><u>To achieve this, consider:</u></p>
          <ul>
            {responseData.trust_score.counterfactual_suggestion.suggested_changes.map((change, idx) => (
              <li key={idx}>
                {change.feature}: currently <strong>{change.current}</strong>,
                ideally <strong>{change.ideal}</strong> (
                <span style={{ color: change.change > 0 ? "green" : "red" }}>
                  {change.change > 0 ? "increase" : "decrease"} by {Math.abs(change.change)}
                </span>)
              </li>
            ))}
          </ul>
        </div>
      )}


      {activeCrop && (
        <div className="modal-overlay" onClick={() => setActiveCrop(null)}>
          <div className="modal-window" onClick={(e) => e.stopPropagation()}>
            <h2>{activeCrop.crop.charAt(0).toUpperCase() + activeCrop.crop.slice(1)}</h2>
            <p><strong>Probability:</strong> {(activeCrop.probability * 100).toFixed(2)}%</p>
            <canvas ref={chartRef} width={400} height={300}></canvas>
            <div className="crop-report">
              <h3>AI-Generated Report</h3>
              <p>{activeCrop.report}</p>
            </div>
          </div>
        </div>
      )}

      <div className="global-importance">
        <h3>🌍 Global Feature Importance</h3>
        <img
          src="http://localhost:3001/results/global_importance.png"
          alt="Global Feature Importance"
          className="global-image"
        />
        <p className="global-text">
          The chart above shows the global SHAP feature importance calculated from the entire training dataset.
          It helps us understand which input features (like rainfall, nitrogen, pH, etc.) have the most influence
          in determining the best crop to grow—<strong>regardless of specific user inputs.</strong>

          <br /><br />
          From the graph, it's evident that <strong>Rainfall</strong> and <strong>Nitrogen</strong> are the most
          critical features in influencing crop recommendation decisions. These factors consistently show strong
          predictive power across all regions and crops in the dataset.

          <br /><br />
          Features like <strong>pH</strong>, <strong>Potassium</strong>, and <strong>Temperature</strong> also
          play substantial roles but may vary more depending on the crop. The broader the SHAP value spread for
          a feature, the more variably it contributes to different crop outcomes.

          <br /><br />
          This global analysis helps agronomists and policymakers prioritize which environmental and soil
          measurements are most important to monitor or improve for consistent agricultural success.
        </p>
      </div>
    </div>
  );
}

export default App;
