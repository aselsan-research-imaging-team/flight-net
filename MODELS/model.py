import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log10
from skimage.metrics import structural_similarity as ssim


def power(input, power_rate):

    out= torch.empty(input.size()).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    for i in range(power_rate.size()[0]):
        out[i] = torch.pow(input[i], power_rate[i])
        out = torch.clip(input, min=0.00001, max=1)
    return out

def mul(input, gain_rate):

    out= input*gain_rate.view(-1,1,1,1)
    return out


class AdaptiveGainCorrection(nn.Module):
    def __init__(self, channel=16):
        super(AdaptiveGainCorrection, self).__init__()

        self.enhance_feature=nn.Sequential(
            nn.Conv2d(3,3,kernel_size=7,stride=2,padding=0),
            nn.Conv2d(3, 3, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(3, channel, kernel_size=3, stride=2, padding=0),
            nn.Conv2d(channel,channel*4,kernel_size=1,padding=0),
            nn.LeakyReLU(0.1),
        )

        self.avr_pool=nn.AdaptiveAvgPool2d(1)
        self.gain_rate=nn.Sequential(
            nn.Linear(channel*4,channel, bias=False),
            nn.LeakyReLU(0.1),
            nn.Linear(channel, 3, bias=False),
            nn.LeakyReLU(0.1),
        )

    def forward(self,x):
        x=self.enhance_feature(x)
        b,c,h,w=x.size()
        x=self.avr_pool(x).view(b,c)
        x=self.gain_rate(x)

        return x

class AdaptiveColorCorrection(nn.Module):
    def __init__(self, channel=16):
        super(AdaptiveColorCorrection, self).__init__()

        self.channel = channel
        self.enhance_feature=nn.Sequential(
            nn.Conv2d(3,3,kernel_size=5,stride=1,padding='same'),
            nn.Conv2d(3, channel, kernel_size=3, stride=1, padding='same'),
            nn.CELU(),
            nn.Conv2d(channel, channel*4, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(channel*4,channel*2,kernel_size=1,padding=0),
            nn.CELU(),
        )

        self.Color_Feature=nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding='same'),
            nn.Conv2d(channel,channel//2, kernel_size=1, stride=1, padding='same'),
            nn.CELU(),
            nn.Conv2d(channel//2,3,kernel_size=1,padding=0),
            nn.ReLU(),
        )

        self.avr_pool=nn.AdaptiveAvgPool2d(1)
        self.gain_rate=nn.Sequential(
            nn.Linear(channel,channel//2, bias=False),
            nn.ReLU(),
            nn.Linear(channel//2, 1, bias=False),
        )
        self.tail=nn.Sequential(
            nn.Conv2d(3,3,1),
            nn.Sigmoid()
            )

    def forward(self, x):
        x1=self.enhance_feature(x)
        spx=torch.split(x1, self.channel, dim=1)

        color_feature=self.Color_Feature(spx[0])

        b,c,h,w=spx[1].size()
        x1=self.avr_pool(spx[1]).view(b, c)
        rate=self.gain_rate(x1)


        out=rate.view(rate.size()[0],1,1,1)*color_feature
        out = self.tail(out)

        return out

#############################################################################################################

class RC_CALayer(nn.Module):
    def __init__(self, channel, reduction=4, dilation=1, kernel_size=1, padding=0):
        super(RC_CALayer, self).__init__()

        self.RC_init=nn.Sequential(
            nn.Conv2d(channel,channel, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(channel,channel, kernel_size=3, padding=1),
            )


        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, padding=padding, bias=True, dilation=1, kernel_size=kernel_size),
            nn.CELU(),
            nn.Conv2d(channel // reduction, channel, padding=padding, bias=True, dilation=1, kernel_size=kernel_size),
            nn.CELU(),
        )
        self.convres=nn.Conv2d(channel,channel,1,1,0)

    def forward(self, x):
        res=x
        x=self.RC_init(x)
        y = self.avg_pool(x)
        y = self.conv_du(y)
        x= x * y
        res=self.convres(res)
        x=x+res
        return x

class CA_Block(nn.Module):
    def __init__(self, channel,Deep=2):
        super(CA_Block, self).__init__()

        self.init=nn.Sequential(
            nn.Conv2d(3, channel, kernel_size=3, padding='same'),
            nn.Tanh())

        modules_RC1=[]
        for i in range(Deep):
            modules_RC1.append(RC_CALayer(channel=channel))
        self.RC1_body=nn.Sequential(*modules_RC1)

        self.convres=nn.Conv2d(channel,channel,1,1,0)
        self.tail=nn.Conv2d(channel,3,3,1,1)

    def forward(self,x):
        x=self.init(x)
        res=x
        x=self.RC1_body(x)
        res=self.convres(res)
        x=x+res
        x=self.tail(x)
        return x
##################################################################################################
class M_ResNet(nn.Module):
    def __init__(self, ):
        super(M_ResNet, self).__init__()
        self.M_ResNet_init=nn.Sequential(
            nn.Conv2d(3,3,kernel_size=5,padding='same'),
            nn.CELU(),
            nn.Conv2d(3,3*4,kernel_size=1,padding='same'),
            nn.CELU(),
            nn.Conv2d(3*4, 8, kernel_size=1, padding='same'),
        )

        self.L1_Split1=nn.Sequential(
            nn.Conv2d(4,4,kernel_size=5,padding='same'),
            nn.CELU(),
            nn.Conv2d(4,4*2,kernel_size=1,padding='same'),
            nn.ReLU(),
            nn.Conv2d(4*2, 4, kernel_size=1, padding='same'),
        )
        self.L1_Split2 =nn.Sequential(
            nn.Conv2d(4,4,kernel_size=3,padding='same'),
            nn.CELU(),
            nn.Conv2d(4,4*2,kernel_size=1,padding='same'),
            nn.CELU(),
            nn.Conv2d(4*2, 4, kernel_size=1, padding='same'),
        )

        self.L1_spxConv = nn.Conv2d(8,3, 3, 1, 1)

        self.L2_init=nn.Sequential(
            nn.Conv2d(3,3,kernel_size=5,padding='same'),
            nn.CELU(),
            nn.Conv2d(3,3*4,kernel_size=1,padding='same'),
            nn.CELU(),
            nn.Conv2d(3*4, 8, kernel_size=1, padding='same'),
        )

        self.L2_Split1=nn.Sequential(
            nn.Conv2d(4,4,kernel_size=5,padding='same'),
            nn.CELU(),
            nn.Conv2d(4,4*2,kernel_size=1,padding='same'),
            nn.ReLU(),
            nn.Conv2d(4*2, 4, kernel_size=1, padding='same'),
        )
        self.L2_Split2 =nn.Sequential(
            nn.Conv2d(4,4,kernel_size=3,padding='same'),
            nn.CELU(),
            nn.Conv2d(4,4*2,kernel_size=1,padding='same'),
            nn.CELU(),
            nn.Conv2d(4*2, 4, kernel_size=1, padding='same'),
        )
        self.L2_spxConv = nn.Conv2d(8, 3, 3, 1, 1)

        self.M_ResNet_Tail=nn.Sequential(
            nn.Conv2d(9,3,kernel_size=7,padding='same'),
            nn.CELU(),
            nn.Conv2d(3,3*4,kernel_size=1,padding='same'),
            nn.CELU(),
            nn.Conv2d(3*4, 3, kernel_size=1, padding='same'),
            nn.Tanh()
        )

    def forward(self, x):
        Res=x
        x=self.M_ResNet_init(x)
        spx = torch.split(x, 4, 1)

        s1=self.L1_Split1(spx[0])
        s2 = self.L1_Split2(spx[1])

        x=torch.cat((s1,s2),1)
        x=self.L1_spxConv(x)
        #
        Res_L1=x

        x=self.L2_init(x)
        spx = torch.split(x, 4, 1)

        s1 = self.L2_Split1(spx[0])
        s2 = self.L2_Split2(spx[1])

        x = torch.cat((s1, s2), 1)
        x = self.L2_spxConv(x)

        x = torch.cat((Res_L1, x), 1)


        x=torch.cat((Res,x),1)
        x=self.M_ResNet_Tail(x)

        return x



class LLIE_Network(nn.Module):
    def __init__(self):
        super(LLIE_Network,self).__init__()

        self.Color_Space_Linearization=AdaptiveColorCorrection()
        self.Gain_Correction = AdaptiveGainCorrection()



        self.FineTune=CA_Block(channel=16)

        self.Denoise=M_ResNet()



        self.init_model()

    def init_model(self):
        # Common practise for initialization.
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    layer.weight, mode="fan_out", nonlinearity="relu"
                )
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, val=0.0)
            elif isinstance(layer, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(layer.weight, val=1.0)
                torch.nn.init.constant_(layer.bias, val=0.0)
            elif isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, val=0.0)

    def forward(self,x):
        input = x
        color_linearization_matrix=self.Color_Space_Linearization(input)
        out = x * color_linearization_matrix

        gain_rate=self.Gain_Correction(out)
        out = out * gain_rate.view(-1, 3, 1, 1)

        x=self.FineTune(out)
        out=out+x

        final_out=self.Denoise(out)


        return final_out



def PSNR(frame_out, gt):
    mse = F.mse_loss(frame_out, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list

def SSIM(input, gt):
    dehaze_list = torch.split(input, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [ssim(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(dehaze_list))]

    return ssim_list