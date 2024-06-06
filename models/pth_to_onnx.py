import argparse
import torch
import torch.onnx
from ruamel import yaml
from transformers import BlipProcessor, BlipForConditionalGeneration
from models.blip import blip_decoder

# 定义模型文件路径
pth_file_path = "F:/Study/22_Research_Group/image_caption/Models/checkpoints/model_base_caption_capfilt_large.pth"
onnx_file_path = "F:/Study/22_Research_Group/image_caption/Models/checkpoints/model_base_caption_capfilt_large.onnx"



# 创建一个示例输入张量
# 假设输入是一个尺寸为(1, 3, 224, 224)的图像
dummy_input = torch.randn(1, 3, 384, 384)


# 定义模型导出函数
def export_onnx_model(dummy_input, onnx_file_path, config):
    # 加载BLIP模型和处理器
    # processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                         vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                         prompt=config['prompt'])  # 创建blip解码器
    # 设置模型为评估模式
    model.eval()
    # 加载预训练的权重文件
    model.load_state_dict(torch.load(pth_file_path, map_location=torch.device('cpu')))
    # 将模型导出为ONNX格式
    torch.onnx.export(
        model,               # 模型
        dummy_input,         # 示例输入
        onnx_file_path,      # ONNX文件路径
        export_params=True,  # 导出模型参数
        opset_version=11,    # ONNX版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=['input'],   # 输入名称
        output_names=['output'],  # 输出名称
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # 动态轴设置
    )
    print(f"模型已成功导出到 {onnx_file_path}")

# 导出模型到ONNX格式
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='F:/Study/22_Research_Group/image_caption/Models/BLIP/configs/caption_coco.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    export_onnx_model(dummy_input, onnx_file_path, config)
    print("onnx文件已输出到：", onnx_file_path)
