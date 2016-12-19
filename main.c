#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Hyperparameters
#define LAYERS 4
#define INPUT_UNITS 3 // includes 1 bias
#define OUTPUT_UNITS 1
int UNITS[LAYERS] = {INPUT_UNITS, 10, 5, OUTPUT_UNITS};
#define LEARNING_RATE 0.01
#define MOMENTUM 0.3
#define MAX_CYCLES 10000

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

#define PI 3.141592654
#define MAX_DATA_LEN 1000

typedef struct data
{
    double *in;
    double *out;
} DATA;

struct data *create_data()
{
    struct data *data = malloc(sizeof(struct data));
    data->in = calloc(INPUT_UNITS + 1, sizeof(double));
    data->out = calloc(OUTPUT_UNITS + 1, sizeof(double));
    return data;
}

typedef struct layer
{
    int num_units;          // number of units in this layer
    double *inputs;         // vector of weighted inputs
    double *outputs;        // vector of outputs
    double *delta;          // vector of delta
    double **weights;       // matrix of weights
    double **updates;       // matrix of updates for momentum
    double **safed_weights; // matrix of weights that have been safed as best
} LAYER;

typedef struct network
{
    struct layer **layers;      // save all layers here (including in and output layer)
    struct layer *input_layer;  // the pointer to the input layer in layers
    struct layer *output_layer; // the pointer to the output layer in layers
    double best_error;          // the best error for which the weights have been saved
} NETWORK;

/**
* Draw from a gaussian distribution with zero mean and variance 1
* Found at: http://c-faq.com/lib/gaussian.html
*/
double gaussrand()
{
    static double V1, V2, S;
    static int phase = 0;
    double X;
    if (phase == 0)
    {
        do
        {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while (S >= 1 || S == 0);
        X = V1 * sqrt(-2 * log(S) / S);
    }
    else
    {
        X = V2 * sqrt(-2 * log(S) / S);
    }
    phase = 1 - phase;
    return X;
}

void initialize_weights(struct network *n)
{
    /*
    double w0[3][4] = {
        { -0.16052908,  0.08038303,  0.21688432,  0.65249966},
        { 0.24712637, -0.64657568,  0.18251471, -0.02146767},
        { 0.09581556,  0.39904124, -0.50837767, -0.62674571}
    };
    double w1[4][1] = {
        {-0.7788337 },
        {-0.45188149},
        {-0.97949992},
        { 0.25871945}
    };*/
    debug("Initializing weights\n");
    // Always connect to the posterior layer
    for (int i = 0; i < LAYERS - 1; i++) // Not from output to posterior
    {
        // Connect all neurons
        for (int j = 0; j < n->layers[i]->num_units; j++)
        {
            for (int k = 0; k < n->layers[i + 1]->num_units; k++)
            {
                n->layers[i]->weights[j][k] = 0.05 * gaussrand(); // from -0.1 to 0.1                
                debug("L%d-N%d to L%d-N%d: weight is %.9lf\n", i, j, i + 1, k, n->layers[i]->weights[j][k]);
            }
        }
    }
}

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
        n->layers[i]->num_units = UNITS[i];
        // Allocate memory for the outputs, delta, weights
        n->layers[i]->outputs = calloc(n->layers[i]->num_units, sizeof(double));
        n->layers[i]->inputs = calloc(n->layers[i]->num_units, sizeof(double));
        n->layers[i]->delta = calloc(n->layers[i]->num_units, sizeof(double));
        n->layers[i]->weights = calloc(n->layers[i]->num_units, sizeof(double *));
        n->layers[i]->updates = calloc(n->layers[i]->num_units, sizeof(double *));
        n->layers[i]->safed_weights = calloc(n->layers[i]->num_units, sizeof(double *));
        // Allocate memory for weights
        if (i < LAYERS - 1) // skip output layer (No weights from output -> ?)
        {
            for (int j = 0; j < n->layers[i]->num_units; j++)
            {
                debug("Layer %d has weight vector for unit %d with %d length\n", i, j, UNITS[i + 1]);
                n->layers[i]->weights[j] = calloc(UNITS[i + 1] + 1, sizeof(double));
                n->layers[i]->updates[j] = calloc(UNITS[i + 1] + 1, sizeof(double));
                n->layers[i]->safed_weights[j] = calloc(UNITS[i + 1] + 1, sizeof(double));
            }
        }
        debug("Created Layer %d with %d neurons\n", i, n->layers[i]->num_units);
    }
    // Initialize the weights
    initialize_weights(n);
    // Save the layer 0 as input layer
    n->input_layer = n->layers[0];
    n->output_layer = n->layers[LAYERS - 1];
    debug("Network created\n");
    return n;
}

double activation_function(double x, int derived)
{
    if (derived)
    {        
        return 1.1439 * (1 - sq(tanh((2.0/3) * x)));
    }
    else
    {     
        return 1.7159 * tanh((2.0/3) * x);
    }
}

void scan_data(struct data **d, int *d_len, struct data **classification, int *c_len)
{
    double x, y;
    int c;
    int phase = 0;
    while (scanf("%lf,%lf,%d", &x, &y, &c) != EOF)
    {
        // phase for reading training data
        if (phase == 0)
        {
            if (x == 0 && y == 0 && c == 0)
            {
                phase++;
                debug("Switching to classification data\n");
                (*d_len)--; // compensate the ++ after the last training data
            }
            else
            {
                d[*d_len] = create_data();
                d[*d_len]->in[0] = x;
                d[*d_len]->in[1] = y;
                d[*d_len]->out[0] = c;
                (*d_len)++;
            }
        }
        else if (phase == 1)
        {
            classification[*c_len] = create_data();
            classification[*c_len]->in[0] = x;
            classification[*c_len]->in[1] = y;
            (*c_len)++;
        }
        debug("Scanned %.3f, %.3f - %d\n", x, y, c);
    }
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

void normalize_input(struct data **d, int d_len, struct data **c, int c_len)
{
    // Normalize all Inputs to (-1, 1)
    for (int i = 0; i < INPUT_UNITS  - 1; i++)
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

void split(struct data **all, int len_all, struct data **training, int *t_len, struct data **validation, int *v_len)
{
    debug("Suffling data\n");
    //shuffle(all, len_all);
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

void set_input(struct network *n, struct data *d)
{
    // Set the output for all neurons in input layer except for bias
    for (int i = 0; i < INPUT_UNITS - 1; i++)
    {
        debug("L%d-N%d: output is %.6lf\n", 0, i, d->in[i]);
        n->input_layer->outputs[i] = d->in[i];
    }    
    n->input_layer->outputs[INPUT_UNITS - 1] = 1;
    debug("L%d-N%d: bias is %.6lf\n", 0, INPUT_UNITS - 1, 1.0);
}

void forward_propagation(struct network *n)
{
    // Propagate the signal from prior to current layer, start at the first hidden layer
    for (int i = 1; i < LAYERS; i++)
    {
        debug("Propagating from L%d to L%d\n", i-1, i);
        struct layer *current = n->layers[i];
        struct layer *prior = n->layers[i - 1];
        // j is for current layer
        for (int j = 0; j < current->num_units; j++)
        {
            double sum = 0.0;
            // k is for the prior layer
            for (int k = 0; k < prior->num_units; k++)
            {
                // calculate the sum of weighted inputs for current layer                
                sum += prior->outputs[k] * prior->weights[k][j]; // weights from k(prior) to j(current)
            }
            // output is the function output of all weighted inputs for that node
            current->inputs[j] = sum;
            current->outputs[j] = activation_function(sum, 0); // 0: non derived
            debug("L%d-N%d: output is %.6lf\n", i, j, current->outputs[j]);
        }
    }
    debug("Finished forward propagation\n");
}

void backwards_propagation(struct network *n, struct data *d)
{
    debug("Starting backwards propagation\n");
    // Calculate the delta in the output layer
    debug("Calculating network error\n");
    for (int i = 0; i < n->output_layer->num_units; i++)
    {
        double error = d->out[i] - n->output_layer->outputs[i]; // Target - Actual
        debug("L%d-N%d: error is %.6lf\n", LAYERS - 1, i, error);        
        n->output_layer->delta[i] = error * activation_function( n->output_layer->outputs[i], 1);
        debug("L%d-N%d: delta is %.6lf\n", LAYERS - 1, i, n->output_layer->delta[i]);
    }
    // Propagate the delta backwards through the net (start from layer before output)
    for (int i = LAYERS - 2; i > 0 ; i--)
    {
        struct layer *posterior = n->layers[i + 1];
        struct layer *current = n->layers[i];
        // Calculate the \delta (=error) for every neuron
        for (int j = 0; j < current->num_units; j++)
        {
            // Take into account the influence from posterior layer
            double weighted_sum_delta = 0.0;
            for (int k = 0; k < posterior->num_units; k++)
            {
                weighted_sum_delta += current->weights[j][k] * posterior->delta[k];
            }
            // save it as the error for this node
            current->delta[j] = activation_function(current->outputs[j], 1) * weighted_sum_delta;
            debug("L%d-N%d: delta is %.6lf\n", i, j, current->delta[j]);
        }
    }
    // Apply the weight change
    for (int i = LAYERS - 1; i > 0; i--)
    {
        struct layer *current = n->layers[i];
        struct layer *prior = n->layers[i - 1];
        // There is no weight to the bias from i-1 to ith layer
        for (int j = 0; j < current->num_units; j++)
        {
            for (int k = 0; k < prior->num_units; k++)
            {
                // save the updates
                prior->updates[k][j] = prior->outputs[k] * current->delta[j] + MOMENTUM * prior->updates[k][j];
                // Weights are installed from prior to current
                prior->weights[k][j] += LEARNING_RATE * prior->updates[k][j];
                debug("L%d-N%d to L%d-N%d: Weight is now at %.6lf\n", i - 1, k, i, j, prior->weights[k][j]);
            }
        }
    }
    debug("Finish backwards propagation\n");
}

void save_weights(struct network *n, double error)
{
    if (fabs(error) <= n->best_error) // only if the error decreased
    {
        debug("Saving weights for error %.6lf\n", error);
        n->best_error = fabs(error);
        for (int i = 0; i < LAYERS - 1; i++) // output does not have weights
        {
            for (int j = 0; j < n->layers[i]->num_units; j++)
            {
                for (int k = 0; k < n->layers[i + 1]->num_units; k++)
                {
                    n->layers[i]->safed_weights[j][k] = n->layers[i]->weights[j][k];
                }
            }
        }
    }
}

void restore_weights(struct network *n)
{
    info("Restoring weights with error %.6lf\n", n->best_error);
    for (int i = 0; i < LAYERS - 1; i++) // output does not have weights
    {
        for (int j = 0; j < n->layers[i]->num_units; j++)
        {
            for (int k = 0; k < n->layers[i + 1]->num_units; k++)
            {
                n->layers[i]->weights[j][k] = n->layers[i]->safed_weights[j][k];
            }
        }
    }
}

double calculate_network_error(struct network *n, struct data **d, int d_len)
{
    // Reset error before validating
    double error = 0.0;
    for (int i = 0; i < d_len; i++)
    {
        set_input(n, d[i]);
        forward_propagation(n);
        // Iterate all neurons in the output layer (but the bias term)
        for (int j = 0; j < n->output_layer->num_units; j++)
        {
            // Calculate the networks error
            error += sq(d[i]->out[j] - n->output_layer->outputs[j]);
        }        
    }
    error /= d_len;
    debug("Network Error is at: %.6lf\n", error);
    return error;
}

void train_network(struct network *n, struct data **d, int d_len)
{
    for (int i = 0; i < d_len; i++)
    {
        debug("Training with data %.6lf, %.6lf - %.6lf\n", d[i]->in[0], d[i]->in[1], d[i]->out[0]);
        // Set input and propagate forward
        set_input(n, d[i]);
        // Start forwards propagation
        forward_propagation(n);
        // And propagate it backwards
        backwards_propagation(n, d[i]);
    }
}

void classify(struct network *n, struct data *d, double *out)
{
    // Set input and propagate forward
    set_input(n, d);
    // Start forwards propagation
    forward_propagation(n);
    // Get the outputs
    for (int i = 0; i < n->output_layer->num_units; i++)
    {
        out[i] = n->output_layer->outputs[i];
    }
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

    // Create Network
    struct network *network = create_network();

    // Normalize Inputs to (-1, 1)
    normalize_input(all, d_len, classification, c_len);

    // Split dataset into training and validation
    split(all, d_len, training, &t_len, validation, &v_len);

    // Calculate the inital Error
    double t_error, v_error, v_last_error;
    v_last_error = v_error = calculate_network_error(network, validation, v_len);
    int cycle = 0;
    while (1)
    {
        //shuffle(training, t_len);
        train_network(network, training, t_len);
        v_last_error = v_error;
        v_error = calculate_network_error(network, validation, v_len);
        t_error = calculate_network_error(network, training, t_len);
        save_weights(network, v_error);

        if (cycle > MAX_CYCLES || v_error <= 0.001 || v_last_error * 10 < v_error)
        {
            restore_weights(network);
            v_error = calculate_network_error(network, validation, v_len);
            t_error = calculate_network_error(network, training, t_len);
            info("Stopped: Validation: %.6lf, \tTraining: %.6lf\n", v_error, t_error);
            break;
        }
        if (cycle % (MAX_CYCLES / 1000) == 0)
        {
            info("%d:\t Validation: %.6lf, \tTraining: %.6lf\n", cycle, v_error, t_error);
        }
        cycle++;
    }
    info("Starting classification\n");
    for (int i = 0; i < c_len; i++)
    {
        classify(network, classification[i], out);
        info("%.3lf\t - ", out[0]);
        output("%s\n",(out[0] > 0 ? "+1" : "-1")); 
    }
}
