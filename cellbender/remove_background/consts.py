"""Constant numbers used in remove-background."""

# Factor by which the mode UMI count of the empty droplet plateau is
# multiplied to come up with a UMI cutoff below which no barcodes are used.
EMPIRICAL_LOW_UMI_TO_EMPTY_DROPLET_THRESHOLD = 0.5

# Default prior for the standard deviation of the LogNormal distribution for
# cell size, used only in the case of the 'simple' model.
SIMPLE_MODEL_D_STD_PRIOR = 0.2

# Probability cutoff for determining which droplets contain cells and which
# are empty.  The droplets n with inferred probability q_n > CELL_PROB_CUTOFF
# are saved as cells in the filtered output file.
CELL_PROB_CUTOFF = 0.5

# Default UMI cutoff. Droplets with UMI less than this will be completely
# excluded from the analysis.
LOW_UMI_CUTOFF = 15

# Default total number of droplets to include in the analysis.
TOTAL_DROPLET_DEFAULT = 25000

# Fraction of the data used for training (versus testing).
TRAINING_FRACTION = 0.9

# Size of minibatch by default.
DEFAULT_BATCH_SIZE = 128

# Fraction of totally empty droplets that makes up each minibatch, by default.
FRACTION_EMPTIES = 0.5

# Prior mean and std in log space for alpha0, the Dirichlet precision for cells.
ALPHA0_PRIOR_LOC = 7.
ALPHA0_PRIOR_SCALE = 2.

# Prior on rho, the swapping fraction: the two concentration parameters alpha and beta.
RHO_ALPHA_PRIOR = 18.  # 1.5
RHO_BETA_PRIOR = 200.  # 20.

# Prior on epsilon, the RT efficiency concentration parameter [Gamma(alpha, alpha)].
EPSILON_PRIOR = 20.  # 1000.
