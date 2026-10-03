import { useNavigate } from "react-router-dom";
import Layout from "../components/Layout.jsx";
import { useSession } from "../state/session.jsx";

export default function Sex() {
  const { gender, setGender } = useSession();
  const navigate = useNavigate();

  // Chosen and confirmed in one step: picking an option is the whole page, so
  // making the user then press Continue would be a click for its own sake.
  const choose = (value) => { setGender(value); navigate("/record"); };

  return (
    <Layout step={1}>
      <h2 className="page">First, a quick question</h2>
      <p className="sub">
        Male and female vocal tracts resonate at different frequencies, so the analysis needs
        to know which to expect. It is also one of the model's inputs.
      </p>

      <div className="choices">
        <button className={`choice${gender === "1" ? " sel" : ""}`} onClick={() => choose("1")}>
          <span className="glyph">♂</span>
          Male
          <small>Formant ceiling 5000 Hz</small>
        </button>
        <button className={`choice${gender === "0" ? " sel" : ""}`} onClick={() => choose("0")}>
          <span className="glyph">♀</span>
          Female
          <small>Formant ceiling 5500 Hz</small>
        </button>
      </div>

      <div className="row" style={{ marginTop: 22 }}>
        <button className="ghost" onClick={() => navigate("/")}>← Back</button>
      </div>
    </Layout>
  );
}
