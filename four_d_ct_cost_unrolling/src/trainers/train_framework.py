from .base_trainer import BaseTrainer
# from utils.misc import log
from ..utils.visualization_utils import  disp_warped_img, disp_training_fig
from ..utils.flow_utils import flow_warp
import numpy as np
from scipy.ndimage.interpolation import zoom as zoom
import torch
import time
from PIL import Image
from ..utils.metrics_utils import AverageMeter



from torch.cuda.amp import autocast 
class TrainFramework(BaseTrainer):
    def __init__(self, train_loader, model, loss_func, args):
        super(TrainFramework, self).__init__(train_loader, model, loss_func, args)

    def _run_one_epoch(self):
        am_batch_time, am_data_time, key_meter_names, key_meters, end = self._init_epoch()
        for i_step, data in enumerate(self.train_loader):
            if i_step > self.args.epoch_size:
                break
            prepared_data = self._prepare_data(data)
            with autocast():
                res_dict = self.model(prepared_data)
                flows, aux = self._post_process_model_output(res_dict)
                loss, l_ph, l_sm, flow_mean, l_mwl, l_cyc, l_kpts = self._compute_loss_terms(data, prepared_data["img1"], prepared_data["img2"], prepared_data["vox_dim"], flows, aux)
            meters = [loss, l_ph, l_sm, l_mwl, l_cyc, l_kpts, flow_mean]
            vals = [m.item() if torch.is_tensor(m) else m for m in meters]
            key_meters.update(vals, prepared_data["img1"].size(0))
            self._optimize(loss)
            self.update_to_tensorboard(am_batch_time, am_data_time, key_meter_names, key_meters, i_step)
            self.i_iter += 1
        self._validate()

    def _compute_loss_terms(self, img1, img2, vox_dim, flows, aux, _, __):
        loss, l_ph, l_sm, flow_mean = self.loss_modules['loss_module'](flows, img1, img2, aux, vox_dim)

        return loss, (l_ph, l_sm, flow_mean)

    def _post_process_model_output(self, res_dict):
        flows12, flows21 = res_dict['flows_fw'][0], res_dict['flows_bk'][0]
        aux12, aux21 = res_dict['flows_fw'][1], res_dict['flows_bk'][1]
            
        flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in zip(flows12, flows21)]
        aux = (aux12, aux21)
        return flows, aux

    def update_to_tensorboard(self, key_meter_names, key_meters):
        if self.rank ==0 and self.i_iter % self.args.record_freq == 0:
            for v, name in zip(key_meters.val, key_meter_names):
                self.summary_writer.add_scalar('Train_' + name, v, self.i_iter)

    def _optimize(self, loss):
        loss = loss.mean()
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
    def _init_epoch(self):
        avg_meter_batch_time = AverageMeter()
        avg_meter_data_time = AverageMeter()
        self.model.train()

        key_meter_names, key_meters = self._init_key_meters()
        end = time.time()
        return avg_meter_batch_time, avg_meter_data_time, key_meter_names, key_meters, end

    def _init_key_meters(self):
        key_meter_names = ['Loss', 'l_ph', 'l_sm', "flow_mean"]
        key_meters = AverageMeter(i=len(key_meter_names), print_precision=4, names=key_meter_names)
        return key_meter_names,key_meters

    def _prepare_data(self, data):
        img1, img2 = [im.unsqueeze(0).float().to(self.rank) for im in [data["template_image"], data["unlabeled_image"]]]
        vox_dim = torch.tensor([[1,1,1.]], dtype=torch.float64).to(self.rank)
        
        data = { 
            "img1": img1,
            "img2": img2,
            "vox_dim": vox_dim,
        }

        return data 
            
    @torch.no_grad()
    def _validate(self, validation_data:dict):
        # self._log.info(f'Running validation on rank {self.rank}..')
        if hasattr(self.args,'dump_disp') and self.args.dump_disp:
            return self._dumpt_disp_fields()
        else:
            if self.args.valid_type == 'synthetic':
                return self.synt_validate(validation_data["synt_validate"])
            elif self.args.valid_type == 'variance_valid':
                return self.variance_validate()
            elif self.args.valid_type == "basic": 
                return self._validate_self(validation_data["validate_self"])

    def _validate_basic(self, validate_data:dict) -> None: # optional - also validate by iou
        if self.i_iter > self.args.save_iter:
            self._save_model(validate_data["avg_loss"], name=self.model_suffix) 

    def _validate_self(self, validate_self_data:dict) -> None:
        self._validate_basic(validate_self_data) 

    
    def synt_validate(self):
        error = 0
        loss = 0

        for i_step, data in enumerate(self.valid_loader):
            img1, img2, flow12 = data
            vox_dim = img1[1].to(self.device)
            img1, img2, flow12 = img1[0].to(self.device), img2[0].to(self.device), flow12[0].to(self.device)
            img1 = img1.unsqueeze(1).float().to(self.device)  # Add channel dimension
            img2 = img2.unsqueeze(1).float().to(self.device)  # Add channel dimension

            output = self.model(img1, img2, vox_dim=vox_dim)

            log(f'flow_size = {output[0].size()}')
            log(f'flow_size = {output[0].shape}')

            flow12_net = output[0].squeeze(0).float().to(self.device)  # Remove batch dimension, net prediction
            epe_map = torch.sqrt(torch.sum(torch.square(flow12 - flow12_net), dim=0)).to(self.device).mean()
            # epe_map = torch.abs(flow12 - flow12_net).to(self.device).mean()
            error += float(epe_map.mean().item())
            log(error)

            _loss, l_ph, l_sm = self.loss_func(output, img1, img2, vox_dim)
            loss += float(_loss.mean().item())

        error /= len(self.valid_loader)
        loss /= len(self.valid_loader)
        print(f'Validation error -> {error}')
        print(f'Validation loss -> {loss}')

        self.writer.add_scalar('Validation Error',
                               error,
                               self.i_epoch)

        self.writer.add_scalar('Validation Loss',
                               loss,
                               self.i_epoch)

        # p_imgs = [plot_image(im.detach().cpu(), show=False) for im in [img1, img2]]
        # p_conc_imgs= np.concatenate((np.concatenate(p_imgs[0][:1]+p_imgs[1][:1]),p_imgs[0][2]+p_imgs[1][2]))[np.newaxis][np.newaxis]
        # p_flows = [plot_flow(fl.detach().cpu(), show=False) for fl in [flow12,flow12_net]]
        # p_flows_conc = np.transpose(np.concatenate((np.concatenate(p_flows[0][:1]+p_flows[1][:1]),)),(2,0,1))[np.newaxis]
        # self.writer.add_images('Valid_Images_{}'.format(self.i_epoch), p_conc_imgs, self.i_epoch)
        # self.writer.add_images('Valid_Flows_{}'.format(self.i_epoch), p_flows_conc, self.i_epoch)

        # p_img_fig = plot_images(img1.detach().cpu(), img2.detach().cpu())
        # p_flo_gt = plot_flow(flow12.detach().cpu())
        # p_flo = plot_flow(flow12_net.detach().cpu())
        # self.writer.add_figure('Valid_Images_{}'.format(self.i_epoch), p_img_fig, self.i_epoch)
        # self.writer.add_figure('Valid_Flows_gt_{}'.format(self.i_epoch), p_flo_gt, self.i_epoch)
        # self.writer.add_figure('Valid_Flows_{}'.format(self.i_epoch), p_flo, self.i_epoch)

        p_valid = plot_validation_fig(img1.detach().cpu(), img2.detach().cpu(), flow12.detach().cpu(),
                                      flow12_net.detach().cpu(), show=False)
        self.writer.add_figure('Valid_Images', p_valid, self.i_epoch)

        return error, loss

    @torch.no_grad()
    def variance_validate(self):
        error_median = 0
        error_mean = 0
        error_short = 0
        max_diff_error = 0
        frame_diff_error = 0
        error_median_box = 0
        error_mean_box = 0
        error_short_box = 0
        max_diff_error_box = 0
        frame_diff_error_box = 0
        loss = 0
        im_h = im_w = 192
        im_d = 64
        flows = torch.zeros([3, im_h, im_w, im_d], device=self.device)
        images_warped = torch.zeros(
            [self.args.variance_valid_len, im_h, im_w, im_d], device=self.device)

        for i_step, data in enumerate(self.valid_loader):

            # Prepare data
            img1, img2, name = data
            vox_dim = img1[1].to(self.device)
            img1, img2 = img1[0].to(self.device), img2[0].to(self.device)
            img1 = img1.unsqueeze(1).float()  # Add channel dimension
            img2 = img2.unsqueeze(1).float()  # Add channel dimension

            if i_step % (self.args.variance_valid_len - 1) == 0:
                image0 = img1
                images_warped[i_step %
                              (self.args.variance_valid_len - 1)] = img1.squeeze(0)
                count = 0

            # Remove batch dimension, net prediction
            res = self.model(img1, img2, vox_dim=vox_dim, w_bk=False)[
                'flows_fw'][0][0].squeeze(0).float()
            flows += res
            images_warped[(i_step % (self.args.variance_valid_len - 1))+1] = flow_warp(img2,
                                                                                   flows.unsqueeze(0))  # im1 recons
            count += 1

            if count == self.args.variance_valid_short_len - 1:
                variance = torch.std(images_warped[:count + 1, :, :, :], dim=0)
                error_short += float(variance.mean().item())
                box_variance = variance[49:148, 49:148, 16:48]
                error_short_box += float(box_variance.mean().item())

                log(error_short)
            if count == self.args.frame_dif+1:
                # calculating variance based only on model
                res = self.model(image0, img2, vox_dim=vox_dim, w_bk=False)[
                                 'flows_fw'][0][0].squeeze(0).float()
                diff_warp_straight = torch.zeros(
                    [2, im_h, im_w, im_d], device=self.device)
                diff_warp_straight[0] = images_warped[0]
                diff_warp_straight[1] = flow_warp(img2, res.unsqueeze(0))
                diff_variance_straight = torch.std(diff_warp_straight, dim=0)
                frame_diff_error += float(diff_variance_straight.median().item())
                box_variance = diff_variance_straight[49:148, 49:148, 16:48]
                frame_diff_error_box += float(box_variance.mean().item())
            if count == self.args.variance_valid_len - 1:
                # calculating max_diff variance
                diff_warp = torch.zeros(
                    [2, im_h, im_w, im_d], device=self.device)
                diff_warp[0] = images_warped[0]
                diff_warp[1] = images_warped[-1]
                diff_variance = torch.std(diff_warp, dim=0)
                max_diff_error += float(diff_variance.mean().item())
                box_variance = diff_variance[49:148, 49:148, 16:48]
                max_diff_error_box += float(box_variance.mean().item())
                
                variance = torch.std(images_warped, dim=0)
                error_median += float(variance.median().item())
                error_mean += float(variance.mean().item())
                box_variance = variance[49:148, 49:148, 16:48]
                error_mean_box += float(box_variance.mean().item())
                error_median_box += float(box_variance.median().item())
                log(error_mean)
                flows = torch.zeros([3, im_h, im_w, im_d], device=self.device)
                count = 0

        max_diff_error /= self.args.variance_valid_sets
        frame_diff_error /= self.args.variance_valid_sets
        error_median /= self.args.variance_valid_sets
        error_mean /= self.args.variance_valid_sets
        error_short /= self.args.variance_valid_sets

        max_diff_error_box /= self.args.variance_valid_sets
        frame_diff_error_box /= self.args.variance_valid_sets
        error_median_box /= self.args.variance_valid_sets
        error_mean_box /= self.args.variance_valid_sets
        error_short_box /= self.args.variance_valid_sets
        print(f'Validation maxDiff error-> {max_diff_error}, Validation error mean -> {error_mean}, Validation error median -> {error_median} Short Validation error -> {error_short}')

        self.writer.add_scalar('Validation Difference_Error', max_diff_error, self.i_epoch)
        self.writer.add_scalar('Validation frame_difference_Error',frame_diff_error, self.i_epoch)
        self.writer.add_scalar('Validation Error(mean)', error_mean,self.i_epoch)
        self.writer.add_scalar('Validation Error(median)',error_median,self.i_epoch)
        self.writer.add_scalar('Validation Short Error',error_short,self.i_epoch)
        self.writer.add_scalar('Validation Difference_Error_box',max_diff_error_box,self.i_epoch)
        self.writer.add_scalar('Validation frame_difference_Error_box',frame_diff_error_box,self.i_epoch)
        self.writer.add_scalar('Validation Error(mean)_box',error_mean_box,self.i_epoch)
        self.writer.add_scalar('Validation Error(median)_box',error_median_box,self.i_epoch)
        self.writer.add_scalar('Validation Short Error_box',error_short_box,self.i_epoch)
 
        p_valid = plot_images(images_warped[0].detach().cpu(), images_warped[-1].detach().cpu(), img2.detach().cpu(), show=False)
        
        self.writer.add_figure('Valid_Images_original', p_valid, self.i_epoch)
        p_dif_valid = plot_images(images_warped[0].detach().cpu(), diff_warp[-1].detach().cpu(), img2.detach().cpu(), show=False)
        p_dif_col = plot_warped_img(images_warped[0].detach().cpu(), images_warped[-1].detach().cpu())
        self.writer.add_figure('Valid_Images_warped', p_dif_col, self.i_epoch)

        return [error_median], ["error_median"]

