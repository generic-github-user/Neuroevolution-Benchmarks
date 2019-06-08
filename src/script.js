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
(async () => {
	for (let i = 0; i < 10; ++i) {
	  const h = await tf_network.fit(inputs, outputs, {
		  batchSize: 5,
		  epochs: 3
	  });
	  console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
	}
})();