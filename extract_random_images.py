import os
import io
import random
import tarfile
from PIL import Image

def extract_random_images_from_tar(tar_path, output_dir, num_images=10):
    """
    从 tar 文件中随机提取指定数量的图片并保存
    
    Args:
        tar_path: tar 文件路径
        output_dir: 输出目录
        num_images: 要提取的图片数量
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开 tar 文件
    with tarfile.open(tar_path, 'r') as tar:
        # 获取所有图片文件（.img_content 文件）
        image_members = []
        
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith('.img_content'):
                image_members.append(member)
        
        print(f"找到 {len(image_members)} 张图片")
        
        if len(image_members) < num_images:
            print(f"警告：tar 文件中只有 {len(image_members)} 张图片，少于请求的 {num_images} 张")
            num_images = len(image_members)
        
        # 随机选择图片
        selected_members = random.sample(image_members, num_images)
        
        # 提取并保存图片
        for i, member in enumerate(selected_members):
            # 提取文件
            file_obj = tar.extractfile(member)
            if file_obj is None:
                continue
            
            # 读取并打开图片
            img_data = file_obj.read()
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            
            # 生成输出文件名（使用对应的 .img_name）
            name_member = None
            base_name = member.name.replace('.img_content', '')
            try:
                name_member = tar.getmember(base_name + '.img_name')
                name_file = tar.extractfile(name_member)
                img_name = name_file.read().decode('utf-8').strip()
                name_file.close()
            except:
                img_name = base_name.split('/')[-1]
            
            # 保存图片（PIL 会自动处理格式，不需要手动加扩展名）
            output_filename = f"sample_{i:03d}_{img_name}.jpg"
            # 避免重复的 .jpg 扩展名
            if output_filename.endswith('.jpg.jpg'):
                output_filename = output_filename[:-4]
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存图片
            img.save(output_path)
            print(f"已保存：{output_path}")
            
            # 关闭文件对象
            file_obj.close()
        
        print(f"\n成功保存 {num_images} 张图片到 {output_dir}")

if __name__ == "__main__":
    # 配置参数
    tar_path = "./rs3_val/rs3-1024-000030.tar"
    output_dir = "./test_image"
    num_images = 10
    
    print(f"正在从 {tar_path} 中随机提取 {num_images} 张图片...")
    extract_random_images_from_tar(tar_path, output_dir, num_images)
    print("完成！")
