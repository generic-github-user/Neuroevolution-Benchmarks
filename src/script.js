const canvas = document.querySelector('#graph');
const ctx = canvas.getContext('2d');

var labels = [];
var neat_losses = [];
var tf_losses = [];
/* tf_losses = [5, 6, 2, 4]; */
/* r = [[5, 6], [2, 4]]; */
const graph = new Chart(ctx, {
	type: 'line',
	data: {
		labels: labels,
		datasets: [
			{
				label: 'TensorFlow.js',
				borderColor: 'rgb(255, 99, 132)',
				data: tf_losses
			},
			{
				label: 'NEAT',
				borderColor: 'rgb(96, 157, 255)',
				data: neat_losses
			}
		]
	},
	options: {
		scales: {
			xAxes: [{
				scaleLabel: {
					display: true,
					labelString: 'Epoch'
				}
			}],
			yAxes: [{
				scaleLabel: {
					display: true,
					labelString: 'Loss'
				}
			}]
		}, 
		animation: {
			duration: 0
		},
		hover: {
			animationDuration: 0
		},
		responsiveAnimationDuration: 0
	}
});

const inputs = tf.randomUniform([10], -10, 10);
const outputs = inputs.square();

var a = inputs.dataSync();
b = [];
// replace this with map?
a.forEach(
	(c) => {
		b.push({
			input: [c],
			output: [c ** 2]
		})
	}
)

var n = neataptic;
var neat_network = new neataptic.Network(1, 1);
var neat_options = {
  mutation: n.methods.mutation.ALL,
  mutationRate: 1,
  clear: true,
  cost: n.methods.cost.MSE,
  iterations: 2
};

const tf_network = tf.sequential();
tf_network.add(tf.layers.dense({units: 5, inputShape: [1], activation: 'relu'}));
tf_network.add(tf.layers.dense({units: 1}));
const tf_optimizer = tf.train.sgd(0.0001);
tf_network.compile({optimizer: tf_optimizer, loss: 'meanSquaredError'});


(async () => {
	for (let i = 0; i < 100; ++i) {
	  const h = await tf_network.fit(inputs, outputs, {
		  batchSize: 5,
		  epochs: 1
	  });
	  r = await neat_network.evolve(b, neat_options);
	  
	  console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
	  labels.push(i);
	  neat_losses.push(r.error);
	  tf_losses.push(h.history.loss[0]);
	  /* neat_network = r.evolved; */
	  console.log(r.evolved)
	  console.log(h.history.loss)
	  graph.update();
	  // wait for page load
	  // use this or width/height values in graph.js
	  drawGraph(neat_network.graph(500, 500), '.draw');
	}
	console.log(r)
})();