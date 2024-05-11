from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

# writer.add_image()
# writer.add_scalar() 

# draw y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)

writer.close()