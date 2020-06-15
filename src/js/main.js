// Import Material UI components.
const {
  Typography,
  Slider,
} = MaterialUI;

var ws;
var interval = null;
var iter_count  = 0;
var inference_done = false;

// Particle filter info.
var DEFAULT_NUM_PARTICLES = 50;
var DEFAULT_NUM_ITERS = 20;
var NUM_PARTICLES = DEFAULT_NUM_PARTICLES;
var NUM_ITERS = DEFAULT_NUM_ITERS;
var PERIOD = 500;

// Shape info
var CIRCLE_RADIUS = 20;
var RECT_WIDTH = 26.6;
var RECT_HEIGHT = 7.6;
var ALPHA = 0.7;

// Display info.

function handleInit(props) {
  ws.send(JSON.stringify({action: 'init', num_particles: NUM_PARTICLES}));

  iter_count = 0;
  inference_done = false;
}

function requestUpdate()
{
  ws.send(JSON.stringify({action: 'update', num_iters: NUM_ITERS}));
  iter_count++;

  if (iter_count >= NUM_ITERS)
  {
    clearInterval(interval);
    inference_done = true;
  }
}

function handleStart(props) {
  // Don't start the interval if this button was already pressed.
  if (inference_done) return;

  interval = setInterval(() => {
    requestUpdate();
  }, PERIOD);
}

function handleEstimate(props) {
  // Don't start the interval if this button was already pressed.
  if (!inference_done) return;

  ws.send(JSON.stringify({action: 'estimate'}));
}

function handleSlider(label, value)
{
  if (label === "iterations") {
    NUM_ITERS = value;
  }
  if (label === "particles") {
    NUM_PARTICLES = value;
  }
}

function InitButton() {
  return (
    <button className="button" onClick={() => handleInit(null)} >
      {"Initialize"}
    </button>
  );
}

function StartButton() {
  return (
    <button className="button" onClick={() => handleStart(null)} >
      {"Start"}
    </button>
  );
}

function EstimateButton() {
  return (
    <button className="button" onClick={() => handleEstimate(null)} >
      {"Estimate"}
    </button>
  );
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

class Circle extends React.Component {
  render() {
    var circleStyle = {
      position: "absolute",
      backgroundColor: this.props.colour,
      top: this.props.y - CIRCLE_RADIUS / 2 + "px",
      left: this.props.x - CIRCLE_RADIUS / 2 + "px",
      borderRadius: "50%",
      width:"20px",
      height:"20px",
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
    var shift_x = -RECT_WIDTH / 2;
    var shift_y = -RECT_HEIGHT / 2;

    var corner_tf = "translate(" + shift_x + "px," + shift_y +"px)";
    var rot = "rotate(" + this.props.theta + "rad)";
    var global_tf = "translate(" + this.props.x + "px," + this.props.y +"px)";

    var tf = global_tf + " " + corner_tf + " " + rot;

    var circleStyle = {
      position: "absolute",
      backgroundColor: this.props.colour,
      transform: tf,
      width:"26.6px",
      height:"7.6px",
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

class Board extends React.Component {
  constructor(props) {
    super(props);

    ws=new WebSocket("ws://localhost:8080/bp");
    ws.onmessage= (evt) => this.handleMessage(evt);

    ws.onopen=function(evt){
      ws.send("Hello");
    }

    this.state = {
      circles: Array(NUM_PARTICLES).fill(null),
      l1: Array(NUM_PARTICLES).fill(null),
      l2: Array(NUM_PARTICLES).fill(null),
      l3: Array(NUM_PARTICLES).fill(null),
      l4: Array(NUM_PARTICLES).fill(null),
      l5: Array(NUM_PARTICLES).fill(null),
      l6: Array(NUM_PARTICLES).fill(null),
      l7: Array(NUM_PARTICLES).fill(null),
      l8: Array(NUM_PARTICLES).fill(null),
      colours: ["#00007f",
                "#0000ff",
                "#007fff",
                "#14ffe2",
                "#7bff7b",
                "#e2ff14",
                "#ff9700",
                "#ff2100",
                "#7f0000"]
    };
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
  }

  render() {
    return (
      <div>
        <div>
          Iteration: {iter_count}
        </div>
        <div className="canvas">
          <img id="obs" src="../../media/obs.png" alt="" />
          <DrawCanvas circles={this.state.circles}
                      l1={this.state.l1}
                      l2={this.state.l2}
                      l3={this.state.l3}
                      l4={this.state.l4}
                      l5={this.state.l5}
                      l6={this.state.l6}
                      l7={this.state.l7}
                      l8={this.state.l8}
                      colours={this.state.colours} />
        </div>
        <div className="controls">
          <InitButton />
          <StartButton />
          <EstimateButton />
          <br/>
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

// const domContainer = document.querySelector('#like_button_container');
// ReactDOM.render(e(LikeButton), domContainer);

ReactDOM.render(
  <Board />,
  document.getElementById('appRoot')
);
