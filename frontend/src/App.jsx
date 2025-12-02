import './App.css'

const App = () => {
  return (
    <div>
      {/* Webcam */}
      <div id="camera-bloc">
        <img
          src="http://localhost:5000/video_feed"
          alt="Video"
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            borderBottomRightRadius: '50px',
            overflow: 'hidden',
            boxShadow: 'inset -20px -20px 49px #bebebe, inset 20px 20px 49px #ffffff'
          }}
        />
      </div>

      {/* Graph */}
      <div
        id="graph-bloc"
        style={{
          position: 'absolute',
          bottom: '10px',
          left: '20px',
          width: '400px',
          height: '350px',
          borderRadius: '50px',
          marginTop: '20px',
          background: 'linear-gradient(45deg, #cacaca, #f0f0f0)',
          boxShadow: '20px -20px 60px #bebebe, -20px 20px 60px #ffffff',
          overflow: 'hidden' 
        }}
      >
        <img
          src="http://localhost:5000/pie_graph"
          alt="Graph"
          style={{ width: '100%', height: '100%', objectFit: 'contain' }}
        />
      </div>
    </div>
  );
};

export default App;
