const inputs = tf.randomUniform([10]);
const outputs = inputs.square();

var n = neataptic
var neat_network = new neataptic.Network(1, 1);

const tf_network = tf.sequential();
tf_network.add(tf.layers.dense({units: 5, inputShape: [1], activation: 'relu'}));
tf_network.add(tf.layers.dense({units: 1}));

tf_network.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
