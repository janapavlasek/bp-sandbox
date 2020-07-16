// Import Material UI components.
const {
  Typography,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  LinearProgress,
} = MaterialUI;

var ws = null;
var interval = null;
var connectInterval = null;
var iter_count  = 0;
var inference_done = false;

// Particle filter info.
var DEFAULT_NUM_PARTICLES = 50;
var DEFAULT_NUM_ITERS = 20;
var NUM_PARTICLES = DEFAULT_NUM_PARTICLES;
var NUM_ITERS = DEFAULT_NUM_ITERS;
var PERIOD = 100;           // ms
var CONNECT_PERIOD = 1000;  // ms
var NEW_MSG = false;

// Algorithm types.
var ALGO_TYPES = {
  PF: {name: "Particle Filter",
       label: "pf",
       render: ( () => renderTheory("algorithms/pf.html") )},
  BP: {name: "Belief Propagation",
       label: "bp",
       render: ( () => renderTheory(null) )},
}

// Shape info
var ALPHA = 0.7;

/*******************
 *  INFERENCE LOOP
 *******************/

function requestUpdate()
{
  // If we haven't gotten a new message yet, wait.
  if (!NEW_MSG && iter_count < NUM_ITERS) return;

  iter_count++;
  NEW_MSG = false;
  ws.send(JSON.stringify({action: 'update', num_iters: NUM_ITERS}));

  if (iter_count >= NUM_ITERS)
  {
    clearInterval(interval);
    inference_done = true;
  }
}

function LinearProgressWithLabel(props) {
  return (
    <div className="progress">
      <div className="progress-bar-text">{`Iteration: ${props.value} / ${NUM_ITERS}`}</div>
      <LinearProgress className="progress-bar" variant="determinate" value={Math.round(100 * props.value / NUM_ITERS)} />
    </div>
  );
}

/*******************
 *   ALGO SELECT
 *******************/

function AlgoForm(props) {
  return (
    <FormControl className="algo-form">
      <InputLabel id="select-algo-label">Algorithm</InputLabel>
      <Select
        labelId="select-algo-label"
        id="select-algo"
        value={props.value}
        onChange={props.onChange}
      >
        <MenuItem value={ALGO_TYPES.PF}>{ALGO_TYPES.PF.name}</MenuItem>
        <MenuItem value={ALGO_TYPES.BP}>{ALGO_TYPES.BP.name}</MenuItem>
      </Select>
    </FormControl>
  );
}

function renderTheory(path) {
  var theory_div = document.getElementById("theory");

  if (path === null) {
    theory_div.innerHTML = "";
  }
  else {
    theory_div.innerHTML = `<object type="text/html" data="${path}" ></object>`;
  }
}

/*******************
 *     BUTTONS
 *******************/

function ConnectionStatus(props) {
  var msg = "Wait";
  var colour = "#ffff00";
  if (props.status === WebSocket.OPEN) {
    msg = "Connected";
    colour = "#00ff00";
  }
  else if (props.status === WebSocket.CLOSED) {
    msg = "Disconnected";
    colour = "#ff0000";
  }

  return (
    <div className="status" style={{backgroundColor: colour}}>
      {msg}
    </div>
  );
}

function handleInit(algo_label) {
  ws.send(
    JSON.stringify({action: 'init',
                    algo: algo_label,
                    num_particles: NUM_PARTICLES})
  );

  iter_count = 0;
  inference_done = false;

  if (interval !== null) {
    clearInterval(interval);
  }
}

function handleStart() {
  // Don't start the interval if this button was already pressed.
  if (inference_done) return;

  interval = setInterval(() => {
    requestUpdate();
  }, PERIOD);
}

function handleEstimate() {
  // Don't start the interval if this button was already pressed.
  if (!inference_done) return;

  ws.send(JSON.stringify({action: 'estimate'}));
}

function Button(props) {
  return (
    <button className="button" onClick={() => props.onClick()} >
      {props.text}
    </button>
  );
}

/*******************
 *     SLIDERS
 *******************/

function handleSlider(label, value)
{
  if (label === "iterations") {
    NUM_ITERS = value;
  }
  if (label === "particles") {
    NUM_PARTICLES = value;
  }
}

function valuetext(value) {
  return '${value}';
}

function DiscreteSlider(props) {
  return (
    <div className="slider">
      <div id="discrete-slider">
        {props.label}
      </div>
      <Slider
        defaultValue={props.default}
        getAriaValueText={valuetext}
        aria-labelledby="discrete-slider"
        valueLabelDisplay="auto"
        step={props.step}
        marks
        min={props.min}
        max={props.max}
        onChangeCommitted={(evt, value) => handleSlider(props.label, value)}
      />
    </div>
  );
}

/*******************
 *     CANVAS
 *******************/

class Circle extends React.Component {
  render() {
    var circleStyle = {
      position: "absolute",
      backgroundColor: this.props.colour,
      top: this.props.y - this.props.r + "px",
      left: this.props.x - this.props.r + "px",
      borderRadius: "50%",
      width: 2 * this.props.r + "px",
      height: 2 * this.props.r + "px",
      opacity:ALPHA,
    };
    return (
      <div style={circleStyle}>
      </div>
    );
  }
}

class Rectangle extends React.Component {
  render() {
    var shift_x = -this.props.w / 2;
    var shift_y = -this.props.h / 2;

    var corner_tf = "translate(" + shift_x + "px," + shift_y +"px)";
    var rot = "rotate(" + this.props.theta + "rad)";
    var global_tf = "translate(" + this.props.x + "px," + this.props.y +"px)";

    var tf = global_tf + " " + corner_tf + " " + rot;

    var circleStyle = {
      position: "absolute",
      backgroundColor: this.props.colour,
      transform: tf,
      width: this.props.w + "px",
      height: this.props.h + "px",
      opacity:ALPHA,
    };
    return (
      <div style={circleStyle}>
      </div>
    );
  }
}

class DrawCanvas extends React.Component {

  renderRoot() {
    var circles = [];
    for (var i = 0; i < this.props.circles.length; i++) {
      if (this.props.circles[i]) {
        circles.push(
          <Circle key={i + "circ"}
                  x={this.props.circles[i][0]}
                  y={this.props.circles[i][1]}
                  r={this.props.circles[i][2]}
                  colour={this.props.colours[0]}/>
        );
      }
    }
    return circles;
  }

  renderRectangle(linkInfo, colour) {
    var rects = [];
    for (var i = 0; i < linkInfo.length; i++) {
      if (linkInfo[i]) {
        rects.push(
          <Rectangle key={i + colour}
                     x={linkInfo[i][0]}
                     y={linkInfo[i][1]}
                     theta={linkInfo[i][2]}
                     w={linkInfo[i][3]}
                     h={linkInfo[i][4]}
                     colour={colour}/>
          );
      }
    }
    return rects;
  }

  render() {
    return (
      <div className="drawcanvas">
        {this.renderRoot()}
        {this.renderRectangle(this.props.l1, this.props.colours[1])}
        {this.renderRectangle(this.props.l2, this.props.colours[2])}
        {this.renderRectangle(this.props.l3, this.props.colours[3])}
        {this.renderRectangle(this.props.l4, this.props.colours[4])}
        {this.renderRectangle(this.props.l5, this.props.colours[5])}
        {this.renderRectangle(this.props.l6, this.props.colours[6])}
        {this.renderRectangle(this.props.l7, this.props.colours[7])}
        {this.renderRectangle(this.props.l8, this.props.colours[8])}
      </div>
    );
  }
}

/*******************
 *   WHOLE PAGE
 *******************/

class SandboxPage extends React.Component {
  constructor(props) {
    super(props);

    // Jet colourmap colours.
    this.colours = ["#00007f",
                    "#0000ff",
                    "#007fff",
                    "#14ffe2",
                    "#7bff7b",
                    "#e2ff14",
                    "#ff9700",
                    "#ff2100",
                    "#7f0000"];

    this.state = {
      connection: WebSocket.CLOSED,
      algo: ALGO_TYPES.PF,
      circles: Array(NUM_PARTICLES).fill(null),
      l1: Array(NUM_PARTICLES).fill(null),
      l2: Array(NUM_PARTICLES).fill(null),
      l3: Array(NUM_PARTICLES).fill(null),
      l4: Array(NUM_PARTICLES).fill(null),
      l5: Array(NUM_PARTICLES).fill(null),
      l6: Array(NUM_PARTICLES).fill(null),
      l7: Array(NUM_PARTICLES).fill(null),
      l8: Array(NUM_PARTICLES).fill(null),
    };

    this.attempting_connection = false;

    // Can't call connect because can't call setState before loading.
    ws = new WebSocket("ws://localhost:8080/bp");
    ws.onmessage = (evt) => this.handleMessage(evt);
    ws.onopen = (evt) => { this.setState({connection: ws.readyState}); };
    ws.onclose = (evt) => this.attemptConnection();
    ws.onerror = (evt) => this.attemptConnection();   // Gets called if the connection fails.
  }

  connect() {
    if (ws !== null) {
      if (ws.readyState !== WebSocket.CLOSED) return;
    }

    ws = new WebSocket("ws://localhost:8080/bp");
    ws.onmessage = (evt) => this.handleMessage(evt);
    ws.onopen = (evt) => {
      this.setState({connection: ws.readyState});
      if (connectInterval !== null) clearInterval(connectInterval);
      this.attempting_connection = false;
    };
    ws.onclose = (evt) => this.attemptConnection();
  }

  attemptConnection() {
    if (this.attempting_connection) return;

    if (this.state.connection !== ws.readyState) {
      this.setState({connection: ws.readyState});
    }

    connectInterval = setInterval(() => {
      this.connect();
    }, CONNECT_PERIOD);

    this.attempting_connection = true;
  }

  handleMessage(msg) {
    var server_msg = JSON.parse(msg.data);
    this.setState({circles: server_msg.circles,
                   l1: server_msg.l1,
                   l2: server_msg.l2,
                   l3: server_msg.l3,
                   l4: server_msg.l4,
                   l5: server_msg.l5,
                   l6: server_msg.l6,
                   l7: server_msg.l7,
                   l8: server_msg.l8});
    NEW_MSG = true;
  }

  handleAlgoSelect(event) {
    this.setState({algo: event.target.value});
  }

  render() {
    return (
      <div>
        <div className="status-wrapper">
          <AlgoForm onChange={(event) => this.handleAlgoSelect(event)} value={this.state.algo}/>
          <ConnectionStatus status={this.state.connection}/>
        </div>
        <div className="canvas">
          <img id="obs" src="app/media/obs.png" alt="" />
          <DrawCanvas circles={this.state.circles}
                      l1={this.state.l1}
                      l2={this.state.l2}
                      l3={this.state.l3}
                      l4={this.state.l4}
                      l5={this.state.l5}
                      l6={this.state.l6}
                      l7={this.state.l7}
                      l8={this.state.l8}
                      colours={this.colours} />
        </div>
        <div className="controls">
          <div className="button-wrapper">
            <Button text="Initialize" onClick={() => handleInit(this.state.algo.label)} />
            <Button text="Start" onClick={() => handleStart()} />
            <Button text="Estimate" onClick={() => handleEstimate()} />
          </div>
          <LinearProgressWithLabel value={iter_count} />
          <DiscreteSlider label="particles" min={10} max={500} default={DEFAULT_NUM_PARTICLES} step={10}/>
          <DiscreteSlider label="iterations" min={5} max={200} default={DEFAULT_NUM_ITERS} step={5}/>
        </div>
      </div>
    );
  }
}

window.onclose=function(){
  ws.close();
}

ReactDOM.render(
  <SandboxPage />,
  document.getElementById('appRoot')
);
