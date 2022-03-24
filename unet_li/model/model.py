from .model_parts import *
from .attention import se_block,cbam_block,eca_block

attention_block = [se_block, cbam_block, eca_block]


class Unet(nn.Module):
    def __init__(self, in_channels, n_classes,phi=0):
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.inconv = DoubleConv(in_channels, 64)
        self.down1 = Down_Block(64, 128)
        self.down2 = Down_Block(128, 256)
        self.down3 = Down_Block(256, 512)
        self.down4 = Down_Block(512, 512)
        self.up1 = Up_Block(1024, 256)
        self.up2 = Up_Block(512, 128)
        self.up3 = Up_Block(256, 64)
        self.up4 = Up_Block(128, 64)
        self.outc = Outc(64, n_classes)
        self.phi = phi
        if 1 <= self.phi and self.phi <= 3:
            self.up1_att      = attention_block[self.phi - 1](512)
            self.up2_att      = attention_block[self.phi - 1](256)
            self.up3_att      = attention_block[self.phi - 1](128)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        if 1 <= self.phi and self.phi <= 3:
            x = self.up1_att(x)
        x = self.up2(x, x3)
        if 1 <= self.phi and self.phi <= 3:
            x = self.up2_att(x)
        x = self.up3(x, x2)
        if 1 <= self.phi and self.phi <= 3:
            x = self.up3_att(x)
        x = self.up4(x, x1)
        return self.outc(x)

# if __name__ == '__main__':
#     path = r'E:\dataset\whu-building\data\whu\train\image\0.tif'
#     image = cv2.imread(path, 1)
#     device = 'cpu'
#     net = Unet(in_channels=3, n_classes=1)
#     net = net.to(device=device)
#     image = image / 255
#     image = numpy.transpose(image, (2, 0, 1))
#     image = numpy.expand_dims(image, axis=0)
#     image = torch.tensor(image).type(torch.FloatTensor)
#     image = image.to(device=device, dtype=torch.float)
#     pred = net(image)
#     print(pred.shape)
