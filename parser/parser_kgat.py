# kgat 모델 학습을 위한 argparse 기반 설정 관리. train.py나 main.py같은 KGAT관련 코드에서 사용
# 명령줄 인자를 읽어서 args 객체를 생성.
# ex. --gpu, --n_epoch, --cf_batch_size, --lr, --use_pretrain 등
import argparse
import torch
from config import OPENAI_API_KEY

def parse_kgat_args():
    parser = argparse.ArgumentParser(description="Run KGAT.")
    parser.add_argument('--gpu', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='GPU number')

    parser.add_argument('--data_name', nargs='?', default='Amazon_data_kg_2018/for_kgat_final',
                        help='Choose a dataset from {}')
    parser.add_argument('--data_dir', nargs='?', default='datasets/',
                        help='Input data path.')

    parser.add_argument('--use_pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model.')
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='datasets/pretrain/',
                        help='Path of learned embeddings.')
    parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/KGAT/model_epoch30.pth',
                        help='Path of stored model.')

    parser.add_argument('--cf_batch_size', type=int, default=1024,
                        help='CF batch size.')
    parser.add_argument('--kg_batch_size', type=int, default=2048,
                        help='KG batch size.')
    parser.add_argument('--test_batch_size', type=int, default=10000,
                        help='Test batch size (the user number to test every batch).')

    parser.add_argument('--embed_dim', type=int, default=64,
                        help='User / entity Embedding size.')
    parser.add_argument('--relation_dim', type=int, default=16,
                        help='Relation Embedding size.')

    parser.add_argument('--laplacian_type', type=str, default='symmetric',
                        help='Specify the type of the adjacency (laplacian) matrix from {symmetric, random-walk}.')
    parser.add_argument('--aggregation_type', type=str, default='gcn',
                        help='Specify the type of the aggregation layer from {gcn, graphsage, bi-interaction}.')
    parser.add_argument('--conv_dim_list', nargs='?', default='[64, 32, 16]',
                        help='Output sizes of every aggregation layer.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]',
                        help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')

    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--cf_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating CF l2 loss.')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--n_epoch', type=int, default=100,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=10,
                        help='Number of epoch for early stopping')

    parser.add_argument('--cf_print_every', type=int, default=1,
                        help='Iter interval of printing CF loss.')
    parser.add_argument('--kg_print_every', type=int, default=1,
                        help='Iter interval of printing KG loss.')
    parser.add_argument('--evaluate_every', type=int, default=1,
                        help='Epoch interval of evaluating CF.')

    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Calculate metric@K when evaluating.')
    
    parser.add_argument('--wandb', action="store_false",
                        help='Whether to use wandb sweep.')
    parser.add_argument('--hops', nargs='+', default=[3],
                        help='Number of hops to discover the KG.')
    parser.add_argument('--num_beams', type=int, default=400,
                        help='Number of beams.')
    parser.add_argument('--max_cand_per_node', type=int, default=50,
                        help='Max candidate per node.')
    
    parser.add_argument('--num_llm_eval', type=int, default=215,
                        help='Number of llm system evaluation.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--num_paths', type=int, default=4,
                        help='Number of paths.')
    parser.add_argument('--OPENAI_API_KEY', type=str, default=OPENAI_API_KEY,
                        help='OPENAI_API_KEY')
    # https://platform.openai.com/docs/pricing
    parser.add_argument('--llm_model', type=str, default='gpt-4o-mini',
                        help='llm model')
    parser.add_argument('--judge_model', type=str, default='gpt-4o',
                        help='judging model, needs a smart LLM')
    parser.add_argument('--verbose', action="store_false",
                        help='whether to log')
    
    # aspect를 위하여!
    parser.add_argument('--review_file', default='Amazon_data_2018/Movies_and_TV_5.json')
    parser.add_argument('--lambda_aspect', type=float, default=0.6,
                    help='Aspect 가중치 λ (0~1)')
    
    
    # model 평가 용
    parser.add_argument('--mode', type=str, default='normal',  # 아직 다른 평가 모드는 구현 전
                        choices=['normal', 'user_cold', 'user_review_cold', 'item_cold', 'item_review_cold'])
    parser.add_argument('--k', type=int, default=2, help='cold-start 임계값 (interaction ≤ k)')

    args = parser.parse_args()

    save_dir = 'trained_model/KGAT/{}/del_low_rating_embed-dim{}_relation-dim{}_{}_{}_{}_lr{}_pretrain{}/'.format(
        args.data_name, args.embed_dim, args.relation_dim, args.laplacian_type, args.aggregation_type,
        '-'.join([str(i) for i in eval(args.conv_dim_list)]), args.lr, args.use_pretrain)
    args.save_dir = save_dir
    

    return args
