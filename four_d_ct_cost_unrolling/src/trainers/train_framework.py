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

    def update_to_tensorboard(self, key_meter_names, key_meters):#, visible_masks, ncc_loss_viz=None):
        if self.rank ==0 and self.i_iter % self.args.record_freq == 0:
            for v, name in zip(key_meters.val, key_meter_names):
                self.summary_writer.add_scalar('Train_' + name, v, self.i_iter)

    def _optimize(self, loss):
        loss = loss.mean()
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        # loss.backward()
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
            elif self.args.valid_type == 'l2r_valid':
                return self._validate_with_gt()
            elif 'cardio_test' in self.args.valid_type: 
                return self._validate_self(validation_data["validate_self"])

    def _validate_self(self, validate_self_data:dict) -> None:
        if self.i_iter > self.args.save_iter:
            self._save_model(validate_self_data["avg_loss"], name=self.model_suffix) 
        return 

    def _validate_batch(self, batch_time, end, i_step, data):
        prepared_data = self._prepare_data(data)

        preds = self.model(prepared_data, w_bk=False)
        flows = preds['flows_fw'][0][0]
        pred_flows = flows.detach().squeeze(0)

        batch_time.update(time.time() - end)

        if i_step % self.args.plot_freq == 0:
            img1_recons = flow_warp(prepared_data["img2"][0].unsqueeze(0), pred_flows.unsqueeze(0))
            p_warped_4 = disp_warped_img(prepared_data["img1"][0].detach().cpu(), img1_recons[0].detach().cpu(), prepared_data["img2"].cpu(), split_index=4)
            p_warped_2 = disp_warped_img(prepared_data["img1"][0].detach().cpu(), img1_recons[0].detach().cpu(), prepared_data["img2"].cpu(), split_index=2)
            p_warped_1_33 = disp_warped_img(prepared_data["img1"][0].detach().cpu(), img1_recons[0].detach().cpu(), prepared_data["img2"].cpu(), split_index=1.33)
            self.summary_writer.add_images('Warped_{}_{}'.format(i_step,"_4"), p_warped_4, self.i_epoch, dataformats='NHWC')
            self.summary_writer.add_images('Warped_{}_{}'.format(i_step,"_2"), p_warped_2, self.i_epoch, dataformats='NHWC')
            self.summary_writer.add_images('Warped_{}_{}'.format(i_step,"_1.33"), p_warped_1_33, self.i_epoch, dataformats='NHWC')

            p_clean_4 = disp_warped_img(prepared_data["img1_original"][0].detach().cpu(), prepared_data["img2_rolled"][0].detach().cpu(), prepared_data["img2_original"].cpu(), split_index=4)
            p_clean_2 = disp_warped_img(prepared_data["img1_original"][0].detach().cpu(), prepared_data["img2_rolled"][0].detach().cpu(), prepared_data["img2_original"].cpu(), split_index=2)
            p_clean_1_33 = disp_warped_img(prepared_data["img1_original"][0].detach().cpu(), prepared_data["img2_rolled"][0].detach().cpu(), prepared_data["img2_original"].cpu(), split_index=1.33)
            self.summary_writer.add_images('Warped_clean_{}_{}'.format(i_step,"_4"), p_clean_4, self.i_epoch, dataformats='NHWC')
            self.summary_writer.add_images('Warped_clean_{}_{}'.format(i_step,"_2"), p_clean_2, self.i_epoch, dataformats='NHWC')
            self.summary_writer.add_images('Warped_clean_{}_{}'.format(i_step,"_1.33"), p_clean_1_33, self.i_epoch, dataformats='NHWC')

            p_valid = disp_training_fig(prepared_data["img1"][0].detach().cpu(), prepared_data["img2"][0].detach().cpu(), pred_flows.cpu())

            self.summary_writer.add_images('Sample_{}'.format(i_step), p_valid, self.i_epoch, dataformats='NCHW')
                             
        end = time.time()
  
    def _dumpt_disp_fields(self):
        dump_path = self.save_root  / "submission" / "task_02"
        if not self.args.docker:
            dump_path.mkdir(parents=True)
        self._log.info(f'rank {self.rank} - Dumping disp fields to {dump_path}')
        batch_time = AverageMeter()
        self.model.eval()
        end = time.time()

        all_error_names = []
        all_error_avgs = []

        error_names = ['TRE', 'LogJacDetStd']
        error_meters = AverageMeter(i=len(error_names))
        for i_step, data in enumerate(self.valid_loader):
            img1, img2 = data['imgs']
            if 'kpts' in data['target'].keys() and 'masks' in data['target'].keys():
                kpts, masks =  data['target']['kpts'], data['target']['masks']
            vox_dim =torch.cat([v[:,None] for v in img1[1]],dim=1).to(self.rank)
            img1, img2 = [im[0].to(self.rank) for im in [img1, img2]]
            img1, img2 = [im.unsqueeze(1).float() for im in [img1, img2]]
            
            # compute output
            flows = self.model(img2, img1, vox_dim=vox_dim, w_bk=False)['flows_fw'][0][0]
            pred_flows = flows.detach().squeeze(0)
            spacing = vox_dim.detach()

            # write_flow_as_nrrd(pred_flows.cpu().numpy(),folderpath="{}/flow_nrrd".format(dump_path), filename="{}_flow.nrrd".format(i_step))

            # measure errors
            if 'kpts' in data['target'].keys() and 'masks' in data['target'].keys():
                es = evaluate_flow(pred_flows, kpts, masks, spacing)
            else:
                es = torch.tensor([-1,-1])
            error_meters.update([l.item() for l in es], img1.size(0))


            # dump flow to file
            disp_fields = resize_flow_tensor(pred_flows, shape=self.args.orig_shape)
            _, case_id = data['target']['case'][0].split('_')
            
            if not self.args.docker:
                filename = f'disp_{int(case_id):04}_{int(case_id):04}.npy'
                np.save(dump_path / filename, disp_fields.squeeze(0).cpu().numpy().astype(np.float32))
            else:
                filename = f'disp_{int(case_id):04}_{int(case_id):04}.npz'
                disp_fields = disp_fields.squeeze(0).cpu().numpy().astype(np.float32)
                disp_x = zoom(disp_fields[0], 0.5, order=2).astype('float16')
                disp_y = zoom(disp_fields[1], 0.5, order=2).astype('float16')
                disp_z = zoom(disp_fields[2], 0.5, order=2).astype('float16')
                disp = np.array((disp_x, disp_y, disp_z))

                # save displacement field
                np.savez_compressed(dump_path / filename, disp)
            
            # warped imgs
            img1_recons = flow_warp(img2, pred_flows.unsqueeze(0))
            p_warped = (disp_warped_img(img1[0].detach().cpu(), img1_recons[0].detach().cpu()).squeeze(0)*255).astype(np.uint8)
            p_valid = disp_training_fig(img1[0].detach().cpu(), img2[0].detach().cpu(), pred_flows.cpu()).squeeze(0).transpose(1,2,0).astype(np.uint8)
            filename = f'warped_{int(case_id):04}_{int(case_id):04}.png'
            Image.fromarray(p_warped).save(dump_path / filename)
            filename = f'valid_{int(case_id):04}_{int(case_id):04}.png'
            Image.fromarray(p_valid).save(dump_path / filename)

            # measure elapsed time
            batch_time.update(time.time() - end)

            if i_step % self.args.print_freq == 0 or i_step == len(self.valid_loader) - 1:
                self._log.info('Test: [{0}/{1}]\t Time {2}\t '.format(i_step, len(self.valid_loader), batch_time) 
                    + ' '.join(map('{:.2f}'.format, error_meters.avg)))
            
            end = time.time()

        all_error_avgs.extend(error_meters.avg)
        all_error_names.extend(['{}'.format(name) for name in error_names])

        return all_error_avgs, all_error_names

    def _validate_with_gt(self):
        batch_time = AverageMeter()
        
        # only use the first GPU to run validation, multiple GPUs might raise error.
        # https://github.com/Eromera/erfnet_pytorch/issues/2#issuecomment-486142360
        #self.model = self.model.module
        self.model.eval()

        end = time.time()

        all_error_names = []
        all_error_avgs = []

        error_names = ['TRE', 'LogJacDetStd']
        error_meters = AverageMeter(i=len(error_names))
        for i_step, data in enumerate(self.valid_loader):
            img1, img2 = data['imgs']
            kpts, masks =  data['target']['kpts'], data['target']['masks']
            vox_dim =torch.cat([v[:,None] for v in img1[1]],dim=1).to(self.rank)
            img1, img2 = [im[0].to(self.rank) for im in [img1, img2]]
            img1, img2 = [im.unsqueeze(1).float() for im in [img1, img2]]
            
            # compute output
            flows = self.model(img1, img2, vox_dim=vox_dim, w_bk=False)['flows_fw'][0][0]
            pred_flows = flows.detach().squeeze(0)
            spacing = vox_dim.detach()

            # measure errors
            es = evaluate_flow(pred_flows, kpts, masks, spacing)
            error_meters.update([l.item() for l in es], img1.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)

            if i_step % self.args.print_freq == 0 or i_step == len(self.valid_loader) - 1:
                self._log.info('Test: [{0}/{1}]\t Time {2}\t '.format(i_step, len(self.valid_loader), batch_time) + ' '.join(map('{:.2f}'.format, error_meters.avg)))
            
            if i_step % self.args.plot_freq == 0:
                # 3d plots
                imgs = [img1, img2]
                figs = plot_imgs_and_lms(imgs, masks, kpts, pred_flows)
                self.summary_writer.add_figure('Valid_{}'.format(i_step), figs, self.i_epoch)
                # warped imgs
                img1_recons = flow_warp(img2[0].unsqueeze(0), pred_flows.unsqueeze(0))
                p_warped = disp_warped_img(img1[0].detach().cpu(), img1_recons[0].detach().cpu(), img2.cpu())
                #self.summary_writer.add_figure('Warped_{}'.format(i_step), p_warped, self.i_epoch)
                self.summary_writer.add_images('Warped_{}'.format(i_step), p_warped, self.i_epoch, dataformats='NHWC')
                # imgs and flow                
                p_valid = disp_training_fig(img1[0].detach().cpu(), img2[0].detach().cpu(), pred_flows.cpu())
                self.summary_writer.add_images('Sample_{}'.format(i_step), p_valid, self.i_epoch, dataformats='NCHW')
                
                #p_valid = plot_images(img1[0].detach().cpu(), img1_recons[0].detach().cpu(),
                #                      img2[0].detach().cpu(), show=False)
                #self.writer.add_figure('Training_Images_warping_difference', p_valid, self.i_epoch)
                #diff_warp = torch.zeros([2, 192, 192, 64], device=self.device)
                #diff_warp[0] = img1[0]
                #diff_warp[1] = img1_recons[0]
                #diff_variance = torch.std(diff_warp, dim=0)
                #diff_error = float(diff_variance.median().item())
                #self.writer.add_scalar('Training error', diff_error,
                #                       self.i_iter)
                

            end = time.time()


        # write error to tf board.
        for value, name in zip(error_meters.avg, error_names):
            self.summary_writer.add_scalar(
                'Valid_{}'.format(name), value, self.i_epoch)

        all_error_avgs.extend(error_meters.avg)
        all_error_names.extend(['{}'.format(name) for name in error_names])

       # self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        # In order to reduce the space occupied during debugging,
        # only the model with more than cfg.save_iter iterations will be saved.
        if self.i_iter > self.args.save_iter:
            self._save_model(all_error_avgs[0], name=self.model_suffix)

        return all_error_avgs, all_error_names
    
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
            # print(name)
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
            # if (i_step + 1) % (self.args.variance_valid_len - 1) == 0:
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
            # torch.cuda.empty_cache()

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
        # loss /= len(self.valid_loader)
        print(
            f'Validation maxDiff error-> {max_diff_error}, Validation error mean -> {error_mean}, Validation error median -> {error_median} Short Validation error -> {error_short}')
        # print(f'Validation loss -> {loss}')

        self.writer.add_scalar('Validation Difference_Error',
                               max_diff_error,
                               self.i_epoch)
        self.writer.add_scalar('Validation frame_difference_Error',
                               frame_diff_error,
                               self.i_epoch)
        self.writer.add_scalar('Validation Error(mean)',
                               error_mean,
                               self.i_epoch)
        self.writer.add_scalar('Validation Error(median)',
                               error_median,
                               self.i_epoch)
        self.writer.add_scalar('Validation Short Error',
                               error_short,
                               self.i_epoch)

        self.writer.add_scalar('Validation Difference_Error_box',
                               max_diff_error_box,
                               self.i_epoch)
        self.writer.add_scalar('Validation frame_difference_Error_box',
                               frame_diff_error_box,
                               self.i_epoch)
        self.writer.add_scalar('Validation Error(mean)_box',
                               error_mean_box,
                               self.i_epoch)
        self.writer.add_scalar('Validation Error(median)_box',
                               error_median_box,
                               self.i_epoch)
        self.writer.add_scalar('Validation Short Error_box',
                               error_short_box,
                               self.i_epoch)
        # self.writer.add_scalar('Validation Loss',
        #                        loss,
        #                        self.i_epoch)

        p_valid = plot_images(images_warped[0].detach().cpu(
        ), images_warped[-1].detach().cpu(), img2.detach().cpu(), show=False)
        # p_valid = plot_image(variance.detach().cpu(), show=False)
        #                               flow12_net.detach().cpu(), show=False)
        self.writer.add_figure('Valid_Images_original', p_valid, self.i_epoch)
        p_dif_valid = plot_images(images_warped[0].detach().cpu(
        ), diff_warp[-1].detach().cpu(), img2.detach().cpu(), show=False)
        p_dif_col = plot_warped_img(images_warped[0].detach().cpu(
        ), images_warped[-1].detach().cpu())
        self.writer.add_figure('Valid_Images_warped', p_dif_col, self.i_epoch)

        return [error_median], ["error_median"]

