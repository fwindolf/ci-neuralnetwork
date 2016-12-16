#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/**
* Hyperparameters
*/
// Network Architecture
#define LAYERS 3
#define INPUT_UNITS 1
#define OUTPUT_UNITS 1
#define OUTPUT_CLASSES 2
#define OUTPUT_METHOD CLASSIFICATION
int UNITS[LAYERS] = {INPUT_UNITS, 30, OUTPUT_UNITS};
// Activation Functions
#define INPUT_FUNCTION TANH
#define OUTPUT_FUNCTION TANH
#define DEFAULT_FUNCTION RELU
// Training Parameters
#define TRAINING_CYCLES 10000
#define LEARNING_RATE 0.01
#define ADAPT_LEARNING_RATE 0
#define MINIBATCH 0
#define MINIBATCH_SIZE 50
#define NORMALIZE_OUTPUT 1
#define MOMENTUM 0
#define WEIGHT_DECAY 0
/**
* Debugging
*/
#define DEBUG 1
#define DEBUG_LEVEL 1 // 0: All, 1: Info
#define output(...)                \
    if (!DEBUG || DEBUG_LEVEL > 0) \
    {                              \
        printf(__VA_ARGS__);       \
    }
#define debug(...)                 \
    if (DEBUG && DEBUG_LEVEL <= 0) \
    {                              \
        printf(__VA_ARGS__);       \
    }
#define info(...)                  \
    if (DEBUG && DEBUG_LEVEL <= 1) \
    {                              \
        printf(__VA_ARGS__);       \
    }
#define sq(x) ((x) * (x))

#define MAX_DATA_LEN 1000
#define DOUBLE_ERROR 0.000001
#define PI 3.141592654

typedef enum activation_function {
    LINEAR,
    TANH,
    GAUSSIAN,
    RELU
} ACTIVATION_FUNCTION;

typedef enum method {
    REGRESSION,
    CLASSIFICATION
} METHOD;

typedef struct data
{
    double *in;
    double *out;
} DATA;

typedef struct layer
{
    int num_units;                     // number of units in this layer
    enum activation_function function; // activation function of this layers units
    double *inputs;                    // vector of weighted inputs
    double *outputs;                   // vector of outputs
    double *errors;                    // vector of errors
    double **weights_change;           // matrix of the weight changes
    double **last_weights_change;      // matrix of the last weight changes for learning rate alteration
    double **weights;                  // matrix of weights
    double **learning_rate;            // matrix of learning rates for every weight
    double **safed_weights;            // matrix of weights that have been safed as best
} LAYER;

typedef struct network
{
    struct layer **layers;      // save all layers here (including in and output layer)
    struct layer *input_layer;  // the pointer to the input layer in layers
    struct layer *output_layer; // the pointer to the output layer in layers
    double error;               // the networks error
    double best_error;          // the best error for which the weights have been saved
} NETWORK;

/**
* Activation function of the neurons
*/
double f(double x, enum activation_function f_a)
{
    if (f_a == LINEAR)
        return x;
    else if (f_a == TANH)
        return 1.7159 * tanh((2.0 / 3) * x);
    else if (f_a == GAUSSIAN)
        return exp(-0.5 * sq(x - 2)); // RBF with \mu = 2 and \beta = 0.5
    else if (f_a == RELU)
        return (x <= 0 ? 0 : x);

    info("Undefined function!\n");
    return 0;
}

/**
* Derivative of the activation function of neurons
*/
double f_(double x, enum activation_function f_a)
{
    if (f_a == LINEAR)
        return 1;
    else if (f_a == TANH)
        return 1.1439 * (1.0 - sq(tanh((2.0 / 3) * x )));
    else if (f_a == GAUSSIAN)
        return (2 - x) * f(x, f_a);
    else if (f_a == RELU)
        return (x <= 0 ? 0 : 1);

    info("Undefined function!\n");
    return 0;
}

/**
* Draw from a gaussian distribution with zero mean and variance 1
* Found at: http://c-faq.com/lib/gaussian.html
*/
double gaussrand(double stdev)
{
    static double U, V;
    static int phase = 0;
    double Z;

    if (phase == 0)
    {
        U = (rand() + 1.) / (RAND_MAX + 2.);
        V = rand() / (RAND_MAX + 1.);
        Z = sqrt(-2 * log(U)) * sin(2 * PI * V);
    }
    else
        Z = sqrt(-2 * log(U)) * cos(2 * PI * V);

    phase = 1 - phase;

    return stdev * Z;
}

/**
* Initialize all weights with random numbers
*/
void initialize_weights(struct network *n)
{
    debug("Initializing weights\n");
    double weight_scale = 1 / pow(10, LAYERS - 1);
    // Always connect to the posterior layer
    for (int i = 0; i < LAYERS - 1; i++) // Not from output to posterior
    {
        // Connect all neurons
        for (int j = 0; j < n->layers[i]->num_units; j++)
        {
            // Dont connect to bias terms
            for (int k = 1; k < n->layers[i + 1]->num_units; k++)
            {
                // From Yan Lecun: Weights should be drawn from a gaussian distribution with stdev m^{-1/2}
                // with m as the number of input connections to that weight
                // also increase weights with factor 10 for every layer
                double stdev =  powf(n->layers[i]->num_units, -0.5) * weight_scale;
                n->layers[i]->weights[j][k] = gaussrand(stdev); // Make weights a bit smaller
                debug("L%d-N%d to L%d-N%d: weight is %.9lf\n", i, j, i + 1, k, n->layers[i]->weights[j][k]);
            }
        }
        weight_scale *= 10;
    }
}

/**
* Save the weights with the highest validation accuracy
*/
void save_weights(struct network *n)
{
    if (fabs(n->error) <= n->best_error) // only if the error decreased
    {
        debug("Saving weights for error %.6lf\n", n->error);
        n->best_error = fabs(n->error);
        for (int i = 0; i < LAYERS - 1; i++) // output does not have weights
        {
            for (int j = 0; j < n->layers[i]->num_units; j++)
            {
                for (int k = 1; k < n->layers[i + 1]->num_units; k++)
                {
                    n->layers[i]->safed_weights[j][k] = n->layers[i]->weights[j][k];
                }
            }
        }
    }
}

/**
* Restore the weights 
*/
void restore_weights(struct network *n)
{
    debug("Restoring weights with error %.6lf\n", n->error);
    for (int i = 0; i < LAYERS - 1; i++) // output does not have weights
    {
        for (int j = 0; j < n->layers[i]->num_units; j++)
        {
            for (int k = 1; k < n->layers[i + 1]->num_units; k++)
            {
                n->layers[i]->weights[j][k] = n->layers[i]->safed_weights[j][k];
            }
        }
    }
}

/**
* Set the inputs to the given data
*/
void set_input(struct network *n, struct data *d)
{
    // Set the output for all neurons in input layer except for bias
    for (int i = 1; i < n->layers[0]->num_units; i++)
    {
        debug("L%d-N%d: output is %.6lf\n", 0, i, d->in[i - 1]);
        n->layers[0]->outputs[i] = d->in[i - 1];
    }
}

/**
* Propagate the signal through the network
*/
void forward_propagation(struct network *n)
{
    // Propagate the signal from prior to current layer, start at the first hidden layer
    for (int i = 1; i < LAYERS; i++)
    {
        // j is for the current layer, so skip bias
        for (int j = 1; j < n->layers[i]->num_units; j++)
        {
            double sum = 0.0;
            // k is for the prior layer, consider input from bias
            for (int k = 0; k < n->layers[i - 1]->num_units; k++)
            {
                // calculate the sum of weighted inputs for layer i
                // from all outputs of prior layer to neuron j times that weight on the connection
                sum += n->layers[i - 1]->outputs[k] * n->layers[i - 1]->weights[k][j];
            }
            // output is the function output of all weighted inputs for that node
            n->layers[i]->inputs[j] = sum;
            n->layers[i]->outputs[j] = f(sum, n->layers[i]->function);
            debug("L%d-N%d: output is %.6lf\n", i, j, n->layers[i]->outputs[j]);
        }
    }
    debug("Finished forward propagation\n");
}

/**
* Propagate the error backwards through the net and adapt the weights
* The error for an individual neuron is calculated by 
* (err_k) * f_(out_i) * 
* f'(out_i) * \sum_j (errors_j * weights_ji)
*/
void backwards_propagation(struct network *n, struct data* d)
{
    debug("Starting backwards propagation\n");
    // Calculate the error in the output layer
    // Iterate all neurons in the output layer (but the bias term)
    for (int i = 1; i < n->output_layer->num_units; i++)
    {
        // save it as the error for this node
        n->output_layer->errors[i] = (d->out[i - 1] - n->output_layer->outputs[i]) * f_(n->output_layer->inputs[i], n->output_layer->function);
        debug("L%d-N%d: error is %.6lf\n", LAYERS - 1, i, n->output_layer->errors[i]);
    }
    // Propagate the error backwards through the net
    for (int i = LAYERS - 2; i >= 1; i--)
    {
        struct layer *posterior = n->layers[i + 1];
        struct layer *current = n->layers[i];
        // Calculate the \delta (=error) for every neuron
        for (int j = 0; j < current->num_units; j++)
        {
            // Take into account the influence from posterior layer
            double weighted_sum_errors = 0.0;
            for (int k = 1; k < posterior->num_units; k++)
            {
                weighted_sum_errors += current->weights[j][k] * posterior->errors[k];
            }
            // save it as the error for this node
            current->errors[j] = f_(current->inputs[j], current->function) * weighted_sum_errors;
            debug("L%d-N%d: error is %.6lf\n", i, j, current->errors[j]);
        }
    }
    for (int i = LAYERS - 1; i > 0; i--)
    {
        struct layer *current = n->layers[i];
        struct layer *prior = n->layers[i - 1];
        // There is no weight to the bias from i-1 to ith layer
        for (int j = 1; j < current->num_units; j++)
        {
            for (int k = 0; k < prior->num_units; k++)
            {
                // Weights are installed from prior to current
                prior->weights_change[k][j] += prior->learning_rate[k][j] * prior->outputs[k] * current->errors[j];
                debug("L%d-N%d to L%d-N%d: Weight change is now at %.6lf\n", i - 1, k, i, j, prior->weights_change[k][j]);
            }
        }
    }
    debug("Finish backwards propagation\n");
}

/**
* Calculate the weight updates for every neuron
*/
void update_weights(struct network *n)
{
    debug("Updating weights\n");
    for (int i = LAYERS - 1; i > 0; i--)
    {
        // Leave out the bias for this layer
        for (int j = 1; j < n->layers[i]->num_units; j++)
        {
            for (int k = 0; k < n->layers[i - 1]->num_units; k++)
            {
                // LeCun: LR should be proportional to sqrt of incoming connections
                if (MINIBATCH && ADAPT_LEARNING_RATE) // If no mini batches the calculation occurs too often
                {
                    // LeCun: Increase the learning rate if weight changes dont oscillate
                    if ((n->layers[i - 1]->last_weights_change[k][j] >= 0) ^ (n->layers[i - 1]->weights_change[k][j] < 0)) // If both are the same
                    {
                        n->layers[i - 1]->learning_rate[k][j] *= 1.05;
                        debug("L%d-N%d to L%d-N%d: Increase learning rate to %.6lf\n", i - 1, k, i, j, n->layers[i - 1]->learning_rate[k][j]);
                    }
                    else
                    {
                        n->layers[i - 1]->learning_rate[k][j] *= 0.5;
                        debug("L%d-N%d to L%d-N%d: Decrease learning rate to %.6lf\n", i - 1, k, i, j, n->layers[i - 1]->learning_rate[k][j]);
                    }
                    debug("L%d-N%d to L%d-N%d: new learning rate is: %.6lf\n", i - 1, k, i, j, n->layers[i - 1]->learning_rate[k][j]);
                }

                // Save old weights
                double old_weight = n->layers[i - 1]->weights[k][j];
                // Apply weight change
                n->layers[i - 1]->weights[k][j] += n->layers[i - 1]->weights_change[k][j];
                n->layers[i - 1]->weights[k][j] += MOMENTUM * n->layers[i - 1]->last_weights_change[k][j]; // add momentum
                n->layers[i - 1]->weights[k][j] -= WEIGHT_DECAY * n->layers[i - 1]->weights[k][j]; // add weight decay (l2 regularization)
                debug("L%d-N%d to L%d-N%d: updating weight to %.6lf from %.6lf\n", i - 1, k, i, j, n->layers[i - 1]->weights[k][j], old_weight);
                // Save the change and reset
                n->layers[i - 1]->last_weights_change[k][j] = n->layers[i - 1]->weights_change[k][j];
                n->layers[i - 1]->weights_change[k][j] = 0;
            }
        }
    }
}

/**
* Compute weight updates based on one training data point
*/
void train_network(struct network *n, struct data *d)
{
    // Set input and propagate forward
    set_input(n, d);
    // Start forwards propagation
    forward_propagation(n);
    // And propagate it backwards
    backwards_propagation(n, d);
    debug("Finished training for that datapoint\n");
}

/**
* Classify the data
*/
void classify(struct network *n, struct data *d, double *out)
{
    // Set input and propagate forward
    set_input(n, d);
    // Start forwards propagation
    forward_propagation(n);
    // Get the outputs
    for (int i = 1; i < n->output_layer->num_units; i++)
    {
        out[i - 1] = n->output_layer->outputs[i];
    }
}

/** 
*
*/
double calculate_network_error(struct network *n, struct data **d, int d_len)
{
    // Reset error before validating
    double error = 0.0;
    for (int i = 0; i < d_len; i++)
    {
        set_input(n, d[i]);
        forward_propagation(n);
        // Iterate all neurons in the output layer (but the bias term)
        for (int j = 1; j < n->output_layer->num_units; j++)
        {
            // Calculate the networks error
            if (OUTPUT_METHOD == REGRESSION)
            {
                error += sq(n->output_layer->outputs[j] - d[i]->out[j - 1]);
            }
            else if (OUTPUT_METHOD == CLASSIFICATION)
            {
                
                error += d[i]->out[j - 1] * log(n->output_layer->outputs[j]);
            }
        }
    }
    error /= d_len;
    debug("Network Error is at: %.6lf\n", error);
    return error;
}

/**
* Calculate the accuracy of the network
*/
double accuracy(struct network *n, struct data **d, int d_len)
{
    int sum = 0;
    double *out = calloc(OUTPUT_UNITS, sizeof(double));
    for (int i = 0; i < d_len; i++)
    {
        int correct = 1;
        classify(n, d[i], out);
        for (int j = 0; j < OUTPUT_UNITS; j++)
        {
            if (OUTPUT_METHOD == REGRESSION)
            {
                if (fabs(out[j] - d[i]->out[j]) >= DOUBLE_ERROR)
                {
                    correct = 0;
                }
            }
            else
            {
                if (out[j] != (d[i]->out[j] < 0 ? -1 : 1))
                {
                    correct = 0;
                }
            }
        }
        sum += correct;
    }
    return ((double)sum / d_len);
}

/** 
* Create and initialize the network
*/
struct network *create_network()
{
    debug("Creating network\n");
    struct network *n = malloc(sizeof(struct network));
    n->layers = calloc(LAYERS, sizeof(struct layer *));
    n->best_error = INFINITY;
    for (int i = 0; i < LAYERS; i++)
    {
        debug("Creating layer %d\n", i);
        // Allocate memory for that exact layer (thus malloc)
        n->layers[i] = malloc(sizeof(struct layer));
        // Save the number of units for that layer
        n->layers[i]->num_units = UNITS[i] + 1;
        n->layers[i]->function = DEFAULT_FUNCTION; // default
        // Allocate memory for the outputs, errors, weights
        n->layers[i]->outputs = calloc(n->layers[i]->num_units + 1, sizeof(double));
        n->layers[i]->inputs = calloc(n->layers[i]->num_units + 1, sizeof(double));
        n->layers[i]->outputs[0] = 1; // bias
        n->layers[i]->errors = calloc(n->layers[i]->num_units, sizeof(double));
        n->layers[i]->weights = calloc(n->layers[i]->num_units, sizeof(double *));
        n->layers[i]->last_weights_change = calloc(n->layers[i]->num_units, sizeof(double *));
        n->layers[i]->weights_change = calloc(n->layers[i]->num_units, sizeof(double *));
        n->layers[i]->safed_weights = calloc(n->layers[i]->num_units, sizeof(double *));
        n->layers[i]->learning_rate = calloc(n->layers[i]->num_units, sizeof(double *));
        if (i < LAYERS - 1) // skip output layer
        {
            for (int j = 0; j < n->layers[i]->num_units; j++)
            {
                debug("Layer %d has weight vector for unit %d with %d length\n", i, j, UNITS[i + 1]);
                n->layers[i]->weights[j] = calloc(UNITS[i + 1] + 1, sizeof(double));
                n->layers[i]->safed_weights[j] = calloc(UNITS[i + 1] + 1, sizeof(double));
                n->layers[i]->last_weights_change[j] = calloc(UNITS[i + 1] + 1, sizeof(double));
                n->layers[i]->weights_change[j] = calloc(UNITS[i + 1] + 1, sizeof(double));
                n->layers[i]->learning_rate[j] = calloc(UNITS[i + 1] + 1, sizeof(double));
                for (int k = 0; k < UNITS[i + 1] + 1; k++)
                {
                    if (ADAPT_LEARNING_RATE)
                    {
                        // Adapt the learning rate proportional to sqrt of incoming connections
                        n->layers[i]->learning_rate[j][k] = sqrt(n->layers[i]->num_units);
                    }
                    else
                    {
                        n->layers[i]->learning_rate[j][k] = sqrt(n->layers[i]->num_units) * LEARNING_RATE;
                    }
                }
            }
        }
        debug("Created Layer %d with %d neurons\n", i, n->layers[i]->num_units);
    }
    // Save the layer 1 as input layer, in case we want linear input layer 
    n->input_layer = n->layers[1];
    n->input_layer->function = INPUT_FUNCTION;
    // Save the layer N as output layer, so we can calculate the error and have a linear output layer
    n->output_layer = n->layers[LAYERS - 1];
    n->output_layer->function = OUTPUT_FUNCTION;

    initialize_weights(n);
    debug("Network created\n");
    return n;
}

/**
* Shuffle the dataset
*/
void shuffle(struct data **d, int len)
{
    struct data *temp;
    for (int i = len - 1; i > 0; i--)
    {
        int random = rand() % (i + 1);
        temp = d[random];
        d[random] = d[i];
        d[i] = temp;
    }
}

/**
* Shuffle and split the data in training and validation data set
*/
void split(struct data **all, int len_all, struct data **training, int *t_len, struct data **validation, int *v_len)
{
    debug("Suffling data\n");
    shuffle(all, len_all);
    (*t_len) = (int)(0.66 * len_all);
    (*v_len) = len_all - *t_len;
    debug("Splitting the dataset of %d entries into training(%d) and validation(%d)\n", len_all, *t_len, *v_len);
    for (int i = 0; i < len_all; i++)
    {
        if (i < *t_len)
        {
            training[i] = all[i];
        }
        else
        {
            validation[i - *t_len] = all[i];
        }
    }
}

/**
* Reserve memory for a data block
*/
struct data *create_data()
{
    struct data *data = malloc(sizeof(struct data));
    data->in = calloc(INPUT_UNITS + 1, sizeof(double));
    data->out = calloc(OUTPUT_UNITS + 1, sizeof(double));
    return data;
}

/**
* Scan the input stream for data in the form of x,y,c\n
* When 0,0,0 occurs, the classification data with form x,y\n starts
*/
void scan_data(struct data **d, int *d_len, struct data **classification, int *c_len)
{
    double x, y;
    int phase = 0;
    while (scanf("%lf,%lf", &x, &y) != EOF)
    {
        // phase for reading training data
        if (phase == 0)
        {
            if (x == 0 && y == 0)
            {
                phase++;
                debug("Switching to classification data\n");
                (*d_len)--; // compensate the ++ after the last training data
            }
            else
            {
                d[*d_len] = create_data();
                d[*d_len]->in[0] = x;
                d[*d_len]->out[0] = y;
                (*d_len)++;
            }
        }
        else if (phase == 1)
        {
            classification[*c_len] = create_data();
            classification[*c_len]->in[0] = x;
            (*c_len)++;
        }
        debug("Scanned %.3f - %.3f\n", x, y);
    }
    //(*c_len)--; // compensate the last c_len++
    debug("Finished reading inputs\n");
}

double min_in(struct data **d, int d_len, int idx)
{
    double m = 100000.0;
    for (int i = 0; i < d_len; i++)
    {
        if (d[i]->in[idx] < m)
            m = d[i]->in[idx];
    }
    return m;
}

double max_in(struct data **d, int d_len, int idx)
{
    double m = -100000.0;
    for (int i = 0; i < d_len; i++)
    {
        if (d[i]->in[idx] > m)
            m = d[i]->in[idx];
    }
    return m;
}

double min_out(struct data **d, int d_len, int idx)
{
    double m = 100000.0;
    for (int i = 0; i < d_len; i++)
    {
        if (d[i]->out[idx] < m)
            m = d[i]->out[idx];
    }
    return m;
}

double max_out(struct data **d, int d_len, int idx)
{
    double m = -100000.0;
    for (int i = 0; i < d_len; i++)
    {
        if (d[i]->out[idx] > m)
            m = d[i]->out[idx];
    }
    return m;
}

void normalize_input(struct data **d, int d_len, struct data **c, int c_len)
{
    // Normalize all Inputs to (-1, 1)
    for (int i = 0; i < INPUT_UNITS; i++)
    {
        double min = min_in(d, d_len, i);
        double max = max_in(d, d_len, i);
        max = fabs(min) > fabs(max) ? fabs(min) : fabs(max);
        debug("Normalizing Input Channel %d with %.6lf\n", i, max);
        for (int j = 0; j < d_len; j++)
        {
            d[j]->in[i] /= max;
        }
        for (int j = 0; j < c_len; j++)
        {
            c[j]->in[i] /= max;
        }
    }
}

double normalize_output(struct data **d, int d_len)
{
    // Normalize all Outputs to (-1, 1)
    for (int i = 0; i < INPUT_UNITS; i++)
    {
        double min = min_out(d, d_len, i);
        double max = max_out(d, d_len, i);
        max = fabs(min) > fabs(max) ? fabs(min) : fabs(max);
        debug("Normalizing Output Channel %d with %.6lf\n", i, max);
        for (int j = 0; j < d_len; j++)
        {
            d[j]->out[i] /= max;
        }
        return max; // TODO make this multiple output 
    }
    return 1.0;
}

/**
* Format a variable length output to fit the format 
* +1 for 1, -1 for -1 with a , separated string for 
* every output
*/
void format_output(int *output)
{
    output("%s", ((output[0] == 1) ? "+1" : "-1"));
    for (int j = 1; j < OUTPUT_UNITS; j++)
    {
        output(",%s", ((output[j] == 1) ? "+1" : "-1"));
    }
    output("\n");
}

int main(void)
{
    // Initialize random
    srand(time(NULL));
    // Create and fill data containers
    debug("Creating data containers\n");
    struct data **all = calloc(MAX_DATA_LEN, sizeof(struct data *));
    struct data **classification = calloc(MAX_DATA_LEN, sizeof(struct data *));
    struct data **training = calloc(MAX_DATA_LEN, sizeof(struct data *));
    struct data **validation = calloc(MAX_DATA_LEN, sizeof(struct data *));
    int d_len = 0, c_len = 0, t_len = 0, v_len = 0;

    // For classification
    double *out = calloc(OUTPUT_UNITS, sizeof(double));

    // Scan the data from the input file
    debug("Scanning data\n");
    scan_data(all, &d_len, classification, &c_len);

    // Create network
    struct network *network = create_network();

    // Find the range of inputs, range of outputs
    // double range_out = max_out(all, d_len, 0) - min_out(all, d_len, 0);
    normalize_input(all, d_len, classification, c_len);
    double out_norm = 1.0;
    if(NORMALIZE_OUTPUT)
    {
        out_norm = normalize_output(all, d_len);
    }


    // Init Training Parameters
    int cycle = 1;
    int did_break = 0;

    // Split dataset into training and validation
    split(all, d_len, training, &t_len, validation, &v_len);

    // Calculate the inital Error
    double t_error, v_error, v_last_error;
    v_last_error = v_error = calculate_network_error(network, validation, v_len);

    // Begin training
    while (cycle <= TRAINING_CYCLES)
    {
        debug("Starting training cycle %d\n", cycle);
        if (MINIBATCH)
        {
            for (int m = 0; m < ((int)t_len / MINIBATCH_SIZE) + 1; m++)
            {
                // Only take the first N samples from training
                shuffle(training, t_len);
                for (int i = 0; i < (t_len < MINIBATCH_SIZE ? t_len : MINIBATCH_SIZE); i++)
                {
                    train_network(network, training[i]);
                }
                update_weights(network);
            }
        }
        else
        {
            shuffle(training, t_len);
            for (int i = 0; i < t_len; i++)
            {
                train_network(network, training[i]);
                update_weights(network);
            }
        }
        // calculate training error
        t_error = calculate_network_error(network, training, t_len);
        v_last_error = v_error;
        v_error = calculate_network_error(network, validation, v_len);
        network->error = v_error; // save this as the current network error
        save_weights(network);
        // Do until we overfit
        if ( v_error < DOUBLE_ERROR) // Rise of 10% in the error -> stop
        {
            if (did_break == 1)
            {
                info("%d: Stopped because validation error increased from %.10lf to %.10lf\n", cycle, v_last_error, v_error);
                break;
            }
            did_break = 1;
        }
        else
        {
            if (did_break == 1)
                did_break = 0;
        }
        if (cycle % (TRAINING_CYCLES / 100) == 0)
        {
            info("%d: Training Error   : %.10lf\n", cycle, t_error);
            info("%d: Validation Error : %.10lf\n", cycle, v_error);
        }
        cycle++;
    }
    // If we ran out of cycles we restore the best weights we had
    if (!did_break)
    {
        info("Restoring best weights (error: %.10lf)\n", network->best_error);
        restore_weights(network);
    }
    if (OUTPUT_METHOD == REGRESSION){
        info("Starting regression\n");
        for (int i = 0; i < c_len; i++)
        {
            classify(network, classification[i], out);
            output("%.6lf\n", out_norm * out[0]); // if NORMALIZE_OUTPUT is set to true, out_norm will carry the normalizing factor
        }
    }
    else {
        info("Starting classification\n");
        for (int i = 0; i < c_len; i++)
        {
            classify(network, classification[i], out);
            if(OUTPUT_FUNCTION == TANH)
                output("%s\n", (out_norm * out[0] > 0 ? "+1" : "-1")); // if NORMALIZE_OUTPUT is set to true, out_norm will carry the normalizing factor
        }
    }
    info("Done!");
    return 0;
}