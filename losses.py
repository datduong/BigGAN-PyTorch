import torch
import torch.nn.functional as F

# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2


def loss_dcgan_gen(dis_fake):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real, wt_real=None, wt_fake=None):
  loss_real = torch.mean(F.relu(1. - dis_real)) # ! average over all samples in batch. ... add weights ? 
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake
# def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
  # loss = torch.mean(F.relu(1. - dis_real))
  # loss += torch.mean(F.relu(1. + dis_fake))
  # return loss


def loss_hinge_gen(dis_fake, wt=None):
  loss = -torch.mean(dis_fake) # ! high loss if we get negative score on Discriminator(img)
  return loss

# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis

# ! try a new loss ? 
# Hinge Loss
def loss_hinge_dis_weighted(dis_fake, dis_real, wt_real, wt_fake):
  loss_real = torch.mean( wt_real * F.relu(1. - dis_real) ) # ! average over all samples in batch. 
  loss_fake = torch.mean( wt_fake * F.relu(1. + dis_fake) )
  return loss_real, loss_fake

def loss_hinge_gen_weighted(dis_fake, wt):
  loss = -torch.mean(wt * dis_fake) # ! high loss if we get negative score on Discriminator(img)
  return loss


generator_loss_weighted = loss_hinge_gen_weighted
discriminator_loss_weighted = loss_hinge_dis_weighted

