import torch
import torch.nn as nn
import clip
import torch.nn.functional as F
from utils import get_prompt_chunks

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AGIQAModel(nn.Module):
    """统一的生成图像评估模型，包含三个评估分支"""
    def __init__(self):
        super().__init__()
        # 加载CLIP模型
        self.clip, _ = clip.load("ViT-B/16", device=device, jit=False)
        self.clip.float()
        self.embed_dim = self.clip.visual.output_dim 
        
        # 冻结模型
        self.freeze_model(fix_rate=0.5)
        
        # 定义属性词库
        self.quality_levels = ["bad", "poor", "fair", "good", "perfect"]
        self.realism_levels = ["unrealistic", "slightly unrealistic", "fairly realistic", "realistic", "highly realistic"]

        # 生成所有可能的联合文本模板
        self.qa_templates = []
        for r in self.realism_levels:
            for q in self.quality_levels:
                self.qa_templates.append(f"A photo of {q} quality that appears {r}.")
        
        # 分数融合权重（动态初始化）
        self.mos_fusion = nn.Parameter(torch.ones(3))  # 三个分数融合权重
        self.alpha = nn.Parameter(torch.tensor(0.0))   

    def freeze_model(self, fix_rate):
        """根据冻结比例冻结模型的前几层"""
        if fix_rate > 0:
            # 计算需要冻结的层数
            text_fix_num = int(12 * fix_rate)  # 文本编码器共12层
            vision_fix_num = int(12 * fix_rate)  # 视觉编码器共12层（ViT-B/32）
            
            # 冻结视觉部分的前N层
            for i, block in enumerate(self.clip.visual.transformer.resblocks):
                if i < vision_fix_num:
                    for param in block.parameters():
                        param.requires_grad = False
    
            # 冻结文本部分的前N层
            for i, block in enumerate(self.clip.transformer.resblocks):
                if i < text_fix_num:
                    for param in block.parameters():
                        param.requires_grad = False

        # 确保 logit_scale 被冻结
        self.clip.logit_scale.requires_grad = False

    def quality_authenticity_branch(self, x, patches):
        """
        联合质量与真实性分支：处理输入图像
        返回两个分数：质量分数和真实性分数
        """
        batch_size = x.size(0)

        # 文本特征
        text_tokens = clip.tokenize(self.qa_templates).to(device)
        text_features = self.clip.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)

        # 拼接整图与patches
        views = torch.cat([x.unsqueeze(1), patches], dim=1)
        views = views.view(-1, *views.shape[-3:])

        # 图像特征
        image_features = self.clip.encode_image(views)
        image_features = F.normalize(image_features, dim=-1)
        image_features = image_features.view(batch_size, 10, -1)

        # 计算图像和文本相似度
        sims = torch.einsum('bip,qp->biq', image_features, text_features)
        full_sim = sims[:, 0, :]
        patch_sims = sims[:, 1:, :].mean(1)
        avg_sim = (full_sim + patch_sims) / 2.0
        joint_probs = F.softmax(avg_sim, dim=-1)
        
        # 重塑概率矩阵为5x5 (quality x realism)
        joint_probs_2d = joint_probs.view(batch_size, 5, 5)  # [batch, quality, realism]
        
        # 边际化得到质量和真实性的概率分布
        quality_probs = joint_probs_2d.sum(dim=2)  # 对realism维度求和得到quality概率
        realism_probs = joint_probs_2d.sum(dim=1)  # 对quality维度求和得到realism概率
        
        # 计算分数 (1-5)
        levels = torch.arange(1, 6, device=quality_probs.device, dtype=quality_probs.dtype).view(1, -1)
        quality_scores = torch.sum(levels * quality_probs, dim=-1)
        realism_scores = torch.sum(levels * realism_probs, dim=-1)
        
        # 将分数缩放到0-5范围
        quality_scores = ((quality_scores - 1.0) / 4.0) * 5.0
        realism_scores = ((realism_scores - 1.0) / 4.0) * 5.0
        
        return quality_scores, realism_scores
    
    def consistency_branch(self, x, patches, dataset_name, prompts):
        """
        一致性分支：使用语义对齐进行细粒度评估
        输入:
            x: (B, C, H, W)
            patches: (B, M, C, H, W) 图像块, M是图像块数量
            prompts: 文本提示列表 (长度 B)
            tokens: 划分的提示片段(数量为N)
        输出: (B,) 一致性分数
        """
        batch_size = x.size(0)
        scores = []

        for i in range(batch_size):
            prompt = prompts[i]
            image = x[i:i+1]  # (1, 3, 224, 224)
            image_patches = patches[i]  # (M, C, H, W)

            # 编码完整图像 (1, embed_dim)
            image_features = self.clip.encode_image(image)   
            image_features = F.normalize(image_features, dim=-1) 

            # 编码完整提示词 (1, embed_dim)
            full_text_tokens = clip.tokenize([prompt]).to(x.device)   
            full_text_features = self.clip.encode_text(full_text_tokens)  
            full_text_features = F.normalize(full_text_features, dim=-1) 

            # 全局相似度 (1,)
            global_sim = F.cosine_similarity(image_features, full_text_features, dim=-1)  

            # 分割提示词
            tokens = get_prompt_chunks(dataset_name, prompt)
            tokens = tokens if tokens else [prompt]

            if len(tokens) == 1:
                score = (global_sim * 5).squeeze()
            else:
                # 编码提示片段 (N, embed_dim)
                segment_tokens = clip.tokenize(tokens).to(x.device)  # (N, 77)
                segment_features = self.clip.encode_text(segment_tokens)  
                segment_features = F.normalize(segment_features, dim=-1) 

                # 批量编码图像块 (M, embed_dim) 
                patch_features = self.clip.encode_image(image_patches)
                patch_features = F.normalize(patch_features, dim=-1)

                # 计算片段-图像块相似度矩阵 (N, M)
                similarity_matrix = torch.matmul(segment_features, patch_features.t())

                # 每个片段最相关的m个图像块取平均
                topk_similarities, _ = similarity_matrix.topk(k=3, dim=1)  # (N, m)
                segment_scores = topk_similarities.mean(dim=1)

                # 计算片段权重 (N,)
                segment_weights = F.cosine_similarity(full_text_features, segment_features.unsqueeze(0), dim=-1).squeeze(0)
                weights = F.softmax(segment_weights, dim=0)

                # 加权求和细粒度分数
                fine_grained_score = (weights * segment_scores).sum()

                # 融合全局和细粒度相似度
                # final_score = 0.7 * global_sim + 0.3 * fine_grained_score
                alpha = torch.sigmoid(self.alpha)
                final_score = alpha * global_sim + (1 - alpha) * fine_grained_score
                score = (final_score * 5).squeeze()
           
            scores.append(score) 
        return torch.stack(scores)  # (B,)
                       
    def forward(self, x, patches, prompts, dataset_name, score_types=None):
        """前向传播"""
        if score_types is None:
            if dataset_name == 'AGIQA-1K':
                score_types = ['MOS']
            elif dataset_name == 'AGIQA-3K':
                score_types = ['mos_quality', 'mos_align']
            elif dataset_name == 'AIGCIQA2023':
                score_types = ['mosz1', 'mosz2', 'mosz3']
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
        
        results = {}
        
        # 获取所有可能需要的分数
        mos_scores, quality_scores, authenticity_scores, consistency_scores = None, None, None, None
        
        # 根据需要计算质量/真实性分数
        if any(score_type in ['MOS', 'mos_quality', 'mosz1', 'mosz2'] for score_type in score_types):
            quality_scores, authenticity_scores = self.quality_authenticity_branch(x, patches)
        
        # 根据需要计算一致性分数
        if any(score_type in ['MOS', 'mos_align', 'mosz3'] for score_type in score_types):
            consistency_scores = self.consistency_branch(x, patches, dataset_name, prompts)

        # 根据映射规则分配分数（按照分数类型顺序）
        for score_type in score_types:
            if score_type == 'MOS':
                mos_weights = torch.softmax(self.mos_fusion, dim=0)
                mos_scores = mos_weights[0] * quality_scores + mos_weights[1] * authenticity_scores + mos_weights[2] * consistency_scores
                results[score_type] = mos_scores
            elif score_type == 'mos_quality':
                results[score_type] = quality_scores
            elif score_type == 'mos_align':
                results[score_type] = consistency_scores
            elif score_type == 'mosz1':
                results[score_type] = quality_scores
            elif score_type == 'mosz2':
                results[score_type] = authenticity_scores
            elif score_type == 'mosz3':
                results[score_type] = consistency_scores
        
        return results