from net import skip,skip_mask
from net.losses import ExclusionLoss, plot_image_grid, StdLoss, GradientLoss,MS_SSIM,tv_loss, SSIM
from net.noise import get_noise
from net.ZipperNet import ZipperNet
from utils.image_io import *
from utils.image_io import EN, cmp_AG, SD, cross_covariance, correlation_coefficients, CC,SF
import numpy as np
import torch
import torch.nn.functional as F
from collections import namedtuple
from net.downsampler import Downsampler


matplotlib.use('TkAgg')

Fusion_result = namedtuple("Fusion_result", ["all_in_focus","all_in_focus_sr", "measure", "out1", "out2"])
data_type = torch.cuda.FloatTensor

output_flag: int = 1

class IR_VIS_Fusion(object):
    def __init__(self, image1_name, image2_name, image1, image2,GT1,GT2,input, plot_during_training=True,
                 show_every=100,
                 num_iter=4000, factor=2,outpath='',
                 original_reflection=None, original_transmission=None):
        # we assume the reflection is static
        self.image1 = image1
        self.image2 = image2

        self.best_epoch = -1

        self.GT1 = np_to_torch(GT1)
        self.GT2 = np_to_torch(GT2)

        self.outpath = outpath

        self.input = input

        self.factor = factor
        self.plot_during_training = plot_during_training
        self.measure = []
        self.loss = []
        self.show_every = show_every
        self.image1_name = image1_name
        self.image2_name = image2_name

        self.num_iter = num_iter
        self.loss_function = None
        self.parameters = None
        self.learning_rate = 0.001
        self.input_depth = 1
        self.reflection_net_input = None
        self.transmission_net_input = None
        self.original_transmission = original_transmission
        self.original_reflection = original_reflection
        self.reflection_net = None
        self.transmission_net = None
        self.total_loss = None
        self.multiscale_loss = None
        self.multiscale = None
        self.reflection_out = None
        self.transmission_out = None
        self.current_result = None
        self.best_result = None
        self._init_all()

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_inputs()
        self._init_parameters()
        self._init_losses()

    def _init_images(self):
        self.image1_torch = np_to_torch(self.image1).type(torch.cuda.FloatTensor)
        self.image2_torch = np_to_torch(self.image2).type(torch.cuda.FloatTensor)

        # self.input_torch = np_to_torch(self.input).type(torch.cuda.FloatTensor)

    def _init_inputs(self):
        data_type = torch.cuda.FloatTensor

        input1 = np_to_pil(self.image1).convert('L')
        input2 = np_to_pil(self.image2).convert('L')
        input1 = pil_to_np(input1.resize((input1.size[0] * self.factor, input1.size[1] * self.factor), Image.BICUBIC))
        input2 = pil_to_np(input2.resize((input2.size[0] * self.factor, input2.size[1] * self.factor), Image.BICUBIC))



        self.input_bicubic_1 = np_to_torch(input1).type(data_type)
        self.input_bicubic_2 = np_to_torch(input2).type(data_type)
        self.alpha_input = (self.image1_torch + self.image2_torch) / 2

    def _init_parameters(self):
        self.parameters = [p for p in self.fusion_net.parameters()]
        self.parameters += [p for p in self.alpha1_net.parameters()]
        self.parameters += [p for p in self.alpha2_net.parameters()]
        self.parameters += [p for p in self.adjust1_net.parameters()]
        self.parameters += [p for p in self.adjust2_net.parameters()]

    def _init_nets(self):
        data_type = torch.cuda.FloatTensor
        pad = 'reflection'
        KERNEL_TYPE = 'lanczos2'
        alpha1_net = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[4, 4, 4, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
        alpha2_net = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[4, 4, 4, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
        adjust1_net = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[4, 4, 4, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
        adjust2_net = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[4, 4, 4, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
        fusion_net=ZipperNet()
        self.fusion_net = fusion_net.type(data_type)
        self.alpha1_net = alpha1_net.type(data_type)
        self.alpha2_net = alpha2_net.type(data_type)
        self.adjust1_net = adjust1_net.type(data_type)
        self.adjust2_net = adjust2_net.type(data_type)

        downsampler = Downsampler(n_planes=self.input_depth, factor=self.factor, kernel_type=KERNEL_TYPE, phase=0.5,
                                  preserve_size=True).type(data_type)
        self.downsampler = downsampler.type(data_type)


    def _init_losses(self):

        self.mse_loss = torch.nn.MSELoss().type(data_type)
        self.l1_loss = torch.nn.L1Loss().type(data_type)
        self.exclusion_loss = ExclusionLoss().type(data_type)
        self.blur_loss = StdLoss().type(data_type)
        self.gradientloss = GradientLoss().type(data_type)
        self.ms_ssim_loss = MS_SSIM(max_val=1)
        #self.ssim = SSIM()

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        print("global approaching... ")
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure3(j)
            self._obtain_current_result(j)
            self._plot_closure(j)
            optimizer.step()
        print("Done! ")


    def _optimization_closure3(self, step):
        reg_noise_std = 0.001

        alpha_input = self.alpha_input + (self.alpha_input.clone().normal_() * reg_noise_std)
        self.all_in_focus_out_sr = self.fusion_net(self.input_bicubic_1,self.input_bicubic_2)
        self.current_alpha1 = self.alpha1_net(alpha_input)
        self.current_alpha2 = self.alpha2_net(alpha_input)

        self.adjust1_ref = self.adjust1_net(alpha_input)[:, :,
                             self.image1_torch.shape[2] // 2:self.image1_torch.shape[2] // 2 + 1,
                             self.image1_torch.shape[3] // 2:self.image1_torch.shape[3] // 2 + 1] * 0.9 + 0.05

        self.adjust2_ref = self.adjust2_net(alpha_input)[:, :,
                           self.image1_torch.shape[2] // 2:self.image1_torch.shape[2] // 2 + 1,
                           self.image1_torch.shape[3] // 2:self.image1_torch.shape[3] // 2 + 1] * 0.9 + 0.05

        self.all_in_focus_out =crop_torch_image(self.downsampler(self.all_in_focus_out_sr))
        # if SR scale=1, delete self.downsampler
        # out_y, out_cb, out_cr = rgb2y_CWH_nol_torch(self.all_in_focus_out)
        out_y = self.all_in_focus_out
        image1 = np_to_pil(torch_to_np(self.image1_torch)).resize(
            (self.all_in_focus_out.shape[3], self.all_in_focus_out.shape[2]), Image.BICUBIC)
        image2 = np_to_pil(torch_to_np(self.image2_torch)).resize(
            (self.all_in_focus_out.shape[3], self.all_in_focus_out.shape[2]), Image.BICUBIC)
        self.image1_torch = np_to_torch(pil_to_np(image1)).type(torch.cuda.FloatTensor)
        self.image2_torch = np_to_torch(pil_to_np(image2)).type(torch.cuda.FloatTensor)

        #image1_y, image1_cb, image1_cr = rgb2y_CWH_nol_torch(self.image1_torch)
        #image2_y, image2_cb, image2_cr = rgb2y_CWH_nol_torch(self.image2_torch)
        image1_y = self.image1_torch
        image2_y = self.image2_torch
        self.input_joint_grads, self.all_in_focus_out_grad, self.input_grad_2 = joint_grad(out_y, image1_y, image2_y)
        eps = 1e-7
        self.total_loss = self.l1_loss((torch.log(torch.abs(self.current_alpha1+eps))) + torch.log(torch.abs(self.all_in_focus_out+eps))*torch.abs(self.adjust1_ref+eps),
                                       torch.log(torch.abs(self.image1_torch+eps)))#+0.2*(1-self.ssim((self.current_alpha1),self.image1_torch))
        #print(torch.mean(self.all_in_focus_out)/torch.mean(self.current_alpha1))
        self.total_loss += self.l1_loss((torch.log(torch.abs(self.current_alpha2+eps)) + torch.log(torch.abs(self.all_in_focus_out+eps)))*torch.abs(self.adjust2_ref+eps),
                                        torch.log(torch.abs(self.image2_torch+eps)))#+0.2*(1-self.ssim((self.current_alpha2),self.image2_torch))

        self.total_loss += 0.2*self.l1_loss(self.input_joint_grads, self.all_in_focus_out_grad)

        self.total_loss += self.mse_loss(torch.mean(self.all_in_focus_out),torch.mean((self.image1_torch+self.image2_torch)/2))


        self.total_loss += 0.25 * self.mse_loss(self.current_alpha1,
                                              torch.tensor([[[[1]]]]).type(torch.cuda.FloatTensor))
        self.total_loss += 0.25 * self.mse_loss(self.current_alpha2,
                                              torch.tensor([[[[1]]]]).type(torch.cuda.FloatTensor))
        self.total_loss += 0.25 * self.mse_loss(self.adjust1_ref,
                                               torch.tensor([[[[0.5]]]]).type(torch.cuda.FloatTensor))
        self.total_loss += 0.25 * self.mse_loss(self.adjust1_ref,
                                               torch.tensor([[[[0.5]]]]).type(torch.cuda.FloatTensor))
        self.total_loss.backward()



    def _obtain_current_result(self, j):
        """
        puts in self.current result the current result.
        also updates the best result
        :return:
        """
        all_in_focus_out = np.clip(torch_to_np(self.all_in_focus_out), 0, 1)
        all_in_focus_sr_out = np.clip(torch_to_np(self.all_in_focus_out_sr), 0, 1)

        out1 = np.clip(torch_to_np(self.current_alpha1), 0, 1)
        out2 = np.clip(torch_to_np(self.current_alpha2), 0, 1)

        Ag = cmp_AG(all_in_focus_sr_out)
        En = EN(all_in_focus_sr_out)
        Sd = SD(all_in_focus_sr_out)
        self.loss.append(self.total_loss.item())
        self.measure.append((Ag+En+Sd)/3)
        self.current_result = Fusion_result(all_in_focus=all_in_focus_out,all_in_focus_sr=all_in_focus_sr_out,measure=(Ag+En+Sd)/3, out1=out1, out2=out2)
        if self.best_result is None or self.best_result.measure < self.current_result.measure:
            if j > 500:
                self.best_result = self.current_result
                self.best_epoch = j

    def _plot_closure(self, step):  # Exclusion {:5f} self.exclusion.item(),
        print('Iteration {:5d}    Loss {:5f} measure: {:f} || adjust1_ref: {:f} adjust2_ref: {:f} '.format(step, self.total_loss.item(),
                                                                   self.current_result.measure,self.adjust1_ref.item(),self.adjust2_ref.item()), '\r', end='')
        if self.plot_during_training and step % self.show_every == self.show_every - 1:
            # plot_image_grid("all_in_focus{}".format(step),
            #                 [self.current_result.reflection, self.current_result.transmission])
            # plot_image_grid("learned_mask_{}".format(step),
            #                 [self.current_result.alpha1, self.current_result.alpha2])
            #save_image("/ir_vis/all_in_focus_{}".format(step), self.current_result.all_in_focus)
            save_image("/ir_vis/fusion_{}".format(step), self.current_result.all_in_focus_sr)

    def finalize(self):
        global output_flag
        outpath = "output/"+self.outpath+"/"
        save_graph("result" + "_measure_"+str(output_flag), self.measure,output_path=outpath)
        save_loss("result" + "_loss_"+str(output_flag), self.loss, output_path=outpath)
        #save_image("result" + "_all_in_focus", self.best_result.all_in_focus,output_path=outpath)
        save_image("result" + "_IR_VIS_fusion"+str(output_flag)+'best_'+str(self.best_epoch), self.best_result.all_in_focus_sr,output_path=outpath)
        save_image("result" + "_label_foreground", self.best_result.out1,output_path=outpath)
        save_image("result" + "_label_background", self.best_result.out2,output_path=outpath)
        print(str(output_flag)+" process done!")

        output_flag += 1



if __name__ == "__main__":

    for i in range(40):
        input1 = prepare_image('./test_data/'+'Test_data_FLIR'+'/IR'+str(i+1)+'.jpg')
        input2 = prepare_image('./test_data/'+'Test_data_FLIR'+'/VIS'+str(i+1)+'.jpg')

        f=2

        input1_pil = np_to_pil(input1).convert('L')
        input1_down = crop_image(input1_pil.resize((input1_pil.size[0] // f, input1_pil.size[1] // f), Image.BICUBIC))
        input1_bicubic = pil_to_np(input1_down.resize((input1_down.size[0] * f, input1_down.size[1] * f), Image.BICUBIC))
        input2_pil = np_to_pil(input2).convert('L')
        input2_down =  crop_image(input2_pil.resize((input2_pil.size[0] // f, input2_pil.size[1] // f), Image.BICUBIC))
        input2_bicubic = pil_to_np(input2_down.resize((input2_down.size[0] * f, input2_down.size[1] * f), Image.BICUBIC))
        step3 = IR_VIS_Fusion('input1', 'input2',pil_to_np(input1_down),pil_to_np(input2_down),input1,input2,
                                       (pil_to_np(input1_down)+pil_to_np(input2_down))/2,plot_during_training=False, num_iter=1000, factor=f,outpath='/ir_vis')
        step3.optimize()
        step3.finalize()
