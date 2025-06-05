from huggingface_hub import snapshot_download
from ..smp import *
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE


FAIL_MSG = 'Failed to obtain answer via API.'


class TUNA_CAP(VideoBaseDataset):

    MD5 = '132831e1e27a54bd8cff6025595670f5'
    TYPE = 'Video-VQA'

    def __init__(self, dataset='TUNA_CAP', nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['TUNA_CAP']

    def prepare_dataset(self, dataset_name='TUNA_CAP', repo_id='friedrichor/TUNA-Bench'):

        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False
            data = load(data_file)
            for video_pth in data['video_path']:
                if not osp.exists(osp.join(pth, video_pth)):
                    return False
            return True

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:

            def unzip_hf_zip(pth):
                import zipfile
                if not osp.exists(osp.join(pth, 'videos')):
                    zip_file = osp.join(pth, 'videos.zip')
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(pth)

            def generate_tsv(pth):
                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if os.path.exists(data_file) and md5(data_file) == self.MD5:
                    return

                data_file = pd.read_parquet(os.path.join(pth, 'tuna_cap/test-00000-of-00001.parquet'))
                data_file = data_file.assign(index=range(len(data_file)))
                data_file['video_path'] = data_file['video'].apply(lambda x: f'./video/{x}.mp4')

                data_file = data_file[['index', 'video', 'video_path', 'visual_characteristic', 'domain', 'question']]

                data_file.to_csv(osp.join(pth, f'{dataset_name}.tsv'), sep='\t', index=False)

            dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            unzip_hf_zip(dataset_path)
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

        return dict(data_file=data_file, root=dataset_path)
    
    def qa_template(self, data):
        question = data['question']
        answer = None
        return question, answer

    def save_video_frames(self, video, video_llm=False):
        vid_path = osp.join(self.data_root, 'video', video + '.mp4')
        import decord
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            nframe = min(len(vid), self.nframe)
            indices = np.linspace(0, len(vid) - 1, nframe, dtype=int)
            frame_paths = self.frame_paths(video, nframe)
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video, len(indices))

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth) and not video_llm:
                    im.save(pth)

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        
        frames, indices, video_info = self.save_video_frames(line['video'], video_llm)

        question, answer = self.qa_template(line)
        message = []
        if video_llm:
            message.append(dict(type='video', value=osp.join(self.data_root, 'video', line['video'] + '.mp4')))
        else:
            for im in frames:
                message.append(dict(type='image', value=im))
        message.append(dict(type='text', value=question))
        return message

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        pass


class TUNA_MCQ(VideoBaseDataset):

    MD5 = '46d0b16289ec80d7da901a46628065fb'
    SYS = "You are an expert in video understanding."
    TYPE = 'Video-MCQ'

    def __init__(self, dataset='TUNA_MCQ', nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.pre_prompt_uniform = """You will be provided with {nframe} separate frames uniformly sampled from a video, the frames are provided in chronological order of the video. \
Analyze these frames and provide the answer to the question about the video content. Answer the multiple-choice question about the video content.
You must use these frames to answer the question; do not rely on any external knowledge or commonsense.
"""
        self.pre_prompt_fps = """You will be provided with separate frames sampled at {fps} fps from a video, the frames are provided in chronological order of the video. \
Analyze these frames and provide the answer to the question about the video content. Answer the multiple-choice question about the video content.
You must use these frames to answer the question; do not rely on any external knowledge or commonsense.
"""
        self.post_prompt = "Answer with the option's letter from the given choices directly."

    @classmethod
    def supported_datasets(cls):
        return ['TUNA_MCQ']

    def prepare_dataset(self, dataset_name='TUNA_MCQ', repo_id='friedrichor/TUNA-Bench'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            
            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False
            data = load(data_file)
            for video_pth in data['video_path']:
                if not osp.exists(osp.join(pth, video_pth)):
                    return False
            return True

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            
            def unzip_hf_zip(pth):
                import zipfile
                if not osp.exists(osp.join(pth, 'videos')):
                    zip_file = osp.join(pth, 'videos.zip')
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(pth)

            def generate_tsv(pth):
                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if os.path.exists(data_file) and md5(data_file) == self.MD5:
                    return

                data_file = pd.read_parquet(os.path.join(pth, 'tuna_mcq/test-00000-of-00001.parquet'))
                data_file = data_file.assign(index=range(len(data_file)))
                data_file['video_path'] = data_file['video'].apply(lambda x: f'./video/{x}.mp4')

                data_file = data_file[['index', 'video_index', 'video', 'video_path', 'visual_characteristic', 
                                       'skill', 'task', 'question', 'answer', 'candidates']]

                data_file.to_csv(osp.join(pth, f'{dataset_name}.tsv'), sep='\t', index=False)

            dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            unzip_hf_zip(dataset_path)
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

        return dict(root=dataset_path, data_file=data_file)

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(eval(data['candidates'])):
            question += f"{chr(ord('A') + idx)}. {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"{chr(ord('A') + answer_idx)}. {answer}"
        return question, answer
    
    def save_video_frames(self, video, video_llm=False):
        vid_path = osp.join(self.data_root, 'video', video + '.mp4')
        import decord
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            num_frames = min(len(vid), self.nframe)
            indices = np.linspace(0, len(vid) - 1, nframe, dtype=int)
            frame_paths = self.frame_paths(video, num_frames)
            self.num_frames = num_frames
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video, len(indices))
            
        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth) and not video_llm:
                    im.save(pth)

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, indices, video_info = self.save_video_frames(line['video'], video_llm)

        question, answer = self.qa_template(line)

        message = [dict(type='text', value=self.SYS, role='system')]
        if video_llm:
            message.append(dict(type='video', value=osp.join(self.data_root, 'video', line['video'] + '.mp4')))
        else:
            for im in frames:
                message.append(dict(type='image', value=im))
        if self.nframe > 0 and self.fps < 0:
            question = self.pre_prompt_uniform.format(nframe=self.nframe).strip() + '\n' + question + '\n' + self.post_prompt.strip()
        else:
            question = self.pre_prompt_fps.format(fps=self.fps).strip() + '\n' + question + '\n' + self.post_prompt.strip()
        message.append(dict(type='text', value=question))
        return message

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.tuna import get_dimension_rating, extract_characters_regex, extract_option

        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'

        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        tgt_file = eval_file.replace('.xlsx', '_rating.json')
        score_file = eval_file.replace('.xlsx', '_score.xlsx')

        if not osp.exists(score_file):
            model = judge_kwargs.get('model', 'exact_matching')
            assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
            print(model)

            if model == 'exact_matching':
                model = None
            elif gpt_key_set():
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            else:
                warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
                model = None
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}
            
            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]
            
            for i, idx in enumerate(data['index']):
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = str(data.loc[data['index'] == idx, 'prediction'].values[0])

                if extract_characters_regex(pred) == '':
                    extract_pred = extract_option(
                        model,
                        data.loc[data['index'] == idx].to_dict(orient='records')[0],
                        'TUNA-MCQ'
                    )
                    data.loc[i, 'score'] = int(extract_pred == ans)
                else:
                    data.loc[i, 'score'] = int(extract_characters_regex(pred) == ans)
            
            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating
