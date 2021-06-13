import argparse
import matplotlib.pyplot as plt
from colorization_by_learning.colorizers import *


class NeuralColorizer:
	def __init__(self, parent=None):
		super(NeuralColorizer, self).__init__()

		# load colorizers
		# colorizer_eccv16 = eccv16(pretrained=True).eval()
		colorizer_siggraph17 = siggraph17(pretrained=True).eval()
		# self.colorizer_eccv16 = colorizer_eccv16.cuda()
		self.colorizer_siggraph17 = colorizer_siggraph17.cuda()

	def colorize(self, img, color = None, mask = None):
		# input_B = torch.Tensor(color)[None, :, :, :]
		# mask_B = torch.Tensor(mask)[None, :, :, :]

		# default size to process images is 256x256
		# grab L channel in both original ("orig") and resized ("rs") resolutions
		(tens_l_orig, tens_l_rs, tens_color, tens_mask) = preprocess_img(img, color, mask, HW=(256,256))
		tens_l_rs = tens_l_rs.cuda()
		tens_color = tens_color.cuda()
		tens_mask = tens_mask.cuda()

		# colorizer outputs 256x256 ab map
		# resize and concatenate to original L channel
		# img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
		# out_img_eccv16 = postprocess_tens(tens_l_orig, self.colorizer_eccv16(tens_l_rs).cpu())
		out_img_siggraph17 = postprocess_tens(tens_l_orig, self.colorizer_siggraph17(tens_l_rs, tens_color, tens_mask).cpu())

		return out_img_siggraph17


# plt.imsave('%s_eccv16.png'%opt.save_prefix, out_img_eccv16)
# plt.imsave('%s_siggraph17.png'%opt.save_prefix, out_img_siggraph17)
#
# plt.figure(figsize=(12,8))
# plt.subplot(2,2,1)
# plt.imshow(img)
# plt.title('Original')
# plt.axis('off')
#
# plt.subplot(2,2,2)
# plt.imshow(img_bw)
# plt.title('Input')
# plt.axis('off')
#
# plt.subplot(2,2,3)
# plt.imshow(out_img_eccv16)
# plt.title('Output (ECCV 16)')
# plt.axis('off')
#
# plt.subplot(2,2,4)
# plt.imshow(out_img_siggraph17)
# plt.title('Output (SIGGRAPH 17)')
# plt.axis('off')
# plt.show()
