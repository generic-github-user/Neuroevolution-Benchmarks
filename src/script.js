const canvas = document.querySelector('#graph');
const ctx = canvas.getContext('2d');

const inputs = tf.randomUniform([10]);
const outputs = inputs.square();

var n = neataptic
var neat_network = new neataptic.Network(1, 1);

const tf_network = tf.sequential();
tf_network.add(tf.layers.dense({units: 5, inputShape: [1], activation: 'relu'}));
tf_network.add(tf.layers.dense({units: 1}));

tf_network.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
var labels = [];
var tf_losses = [];
(async () => {
	for (let i = 0; i < 10; ++i) {
	  const h = await tf_network.fit(inputs, outputs, {
		  batchSize: 5,
		  epochs: 1
	  });
	  console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
	  labels.push(i);
	  tf_losses.push(h.history.loss[0]);
	  console.log(h.history.loss)
	}
})().then(
	() => {
		/* tf_losses = [5, 6, 2, 4]; */
		/* r = [[5, 6], [2, 4]]; */
		const graph = new Chart(ctx, {
			type: 'line',
			data: {
				labels: labels,
				datasets: [{
					label: 'My First dataset',
					backgroundColor: 'rgb(255, 99, 132)',
					borderColor: 'rgb(255, 99, 132)',
					data: tf_losses
				}]
			},
			options: {
				animation: {
					duration: 0 // general animation time
				},
				hover: {
					animationDuration: 0 // duration of animations when hovering an item
				},
				responsiveAnimationDuration: 0 // animation duration after a resize
			}
		});
		graph.update();
	}
)