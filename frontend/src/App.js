import { useState } from "react";

function App() {
  const [formData, setFormData] = useState({
    area: "",
    bedrooms: "",
    bathrooms: "",
    stories: "",
    mainroad: "yes",
    guestroom: "no",
    basement: "no",
    hotwaterheating: "no",
    airconditioning: "yes",
    parking: "",
    prefarea: "yes",
    furnishingstatus: "furnished"
  });

  const [prediction, setPrediction] = useState(null);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch("http://127.0.0.1:5050/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          ...formData,
          area: Number(formData.area),
          bedrooms: Number(formData.bedrooms),
          bathrooms: Number(formData.bathrooms),
          stories: Number(formData.stories),
          parking: Number(formData.parking)
        })
      });

      const data = await response.json();
      setPrediction(data.predicted_price);

    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div>
      <h2>House Price Predictor</h2>

      <form onSubmit={handleSubmit}>
        <input name="area" placeholder="Area" onChange={handleChange} />
        <input name="bedrooms" placeholder="Bedrooms" onChange={handleChange} />
        <input name="bathrooms" placeholder="Bathrooms" onChange={handleChange} />
        <input name="stories" placeholder="Stories" onChange={handleChange} />
        <input name="parking" placeholder="Parking" onChange={handleChange} />

        <button type="submit">Predict</button>
      </form>

      {prediction && (
        <h3>Predicted Price: ${prediction.toLocaleString()}</h3>
      )}
    </div>
  );
}

export default App;