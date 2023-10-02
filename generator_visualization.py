#%%
import torch
from torch.utils.tensorboard import SummaryWriter
from models import Generator,Discriminator
#%%
# Set device to GPU if available, otherwise use CPU
device = torch.device('cpu')


# Create dummy inputs for the Generator and Discriminator
dummy_input_G = torch.randn(1, 3, 256, 256).to(device)
dummy_input_D = torch.randn(1, 3, 256, 256).to(device)

# Load the Generator and Discriminator models
generator = Generator(input_nc=3, output_nc=3, n_residual_blocks=9).to(device)
discriminator = Discriminator(input_nc=3).to(device)

# Specify the log directory where the tensorboard data will be stored
log_dir = "logs"

# Create a SummaryWriter to write the model graph to TensorBoard
writer = SummaryWriter(log_dir=log_dir)

# Add the dummy inputs to the model graphs

#writer.add_graph(generator, dummy_input_G)
writer.add_graph(discriminator, dummy_input_D)    
    


#%%
# %tensorboard --logdir=logs
# %%
