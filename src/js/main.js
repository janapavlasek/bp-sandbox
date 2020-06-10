// 'use strict';

// const e = React.createElement;

// class LikeButton extends React.Component {
//   constructor(props) {
//     super(props);
//     this.state = { liked: false };
//   }

//   render() {
//     if (this.state.liked) {
//       return 'You liked this.';
//     }

//     return e(
//       'button',
//       { onClick: () => this.setState({ liked: true }) },
//       'Like'
//     );
//   }
// }

var ws;
var NUM_PARTICLES = 10;
var SERVER_MSG = null;
var NEW_MSG = false;

function handleInit(props) {
  ws.send({action: 'init', num_particles: NUM_PARTICLES});
}

function InitButton() {
  return (
    <button className="button" onClick={() => handleInit(null)} >
      {"Initialize"}
    </button>
  );
}

class Circle extends React.Component {
  render() {
    var circleStyle = {
      position: "absolute",
      backgroundColor: this.props.colour,
      top: this.props.y - 10 + "px",
      left: this.props.x - 10 + "px",
      borderRadius: "50%",
      width:"20px",
      height:"20px",
      opacity:0.7,
    };
    return (
      <div style={circleStyle}>
      </div>
    );
  }
}

class Rectangle extends React.Component {
  render() {
    var circleStyle = {
      position: "absolute",
      backgroundColor: this.props.colour,
      top: this.props.y - 10 + "px",
      left: this.props.x - 10 + "px",
      transform: "rotate(" + this.props.theta + "deg)",
      width:"40px",
      height:"20px",
      opacity:0.7,
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
    console.log("render circles");
    console.log(circles);
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
    SERVER_MSG = JSON.parse(msg.data);
    this.setState({circles: SERVER_MSG.circles,
                   l1: SERVER_MSG.l1,
                   l2: SERVER_MSG.l2,
                   l3: SERVER_MSG.l3,
                   l4: SERVER_MSG.l4,
                   l5: SERVER_MSG.l5,
                   l6: SERVER_MSG.l6,
                   l7: SERVER_MSG.l7,
                   l8: SERVER_MSG.l8});
  }

  render() {
    return (
      <div>
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
        <InitButton />
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
