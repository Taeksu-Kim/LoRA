# LoRA(LoRA: Low-Rank Adaptation of Large Language Models)

paper : https://arxiv.org/pdf/2106.09685v2.pdf   
official github : https://github.com/microsoft/LoRA   
papers with code : https://paperswithcode.com/paper/lora-low-rank-adaptation-of-large-language   

## Files 

NSMC_test_LoRA.ipynb : NSMC(네이버 영화 리뷰 긍부정 감성분류 데이터셋)에 koelectra를 적용하여 LoRA를 적용해본 파일

convert_lora.py : 기존 모델에 LoRA를 적용하는 코드가 들어있는 파일

## Details

LoRA는 기존의 linear, embedding, 2d convolution layer들을 LoRA의 linear, embedding, 2d convolution layer로 대체하는 방식입니다. 위의 conver_lora.py 파일에서는 2d convolution은 제외하고, PLM에서 자주 사용되는 embedding, linear만 적용하였습니다.

LoRA의 기본 아이디어는 모델의 전체 구조는 똑같이 유지합니다. 하지만 linear, embedding 등 LoRA를 적용할 layer는 freeze시키고, 해당 각 각 원본 layer의 파라미터 차원을 줄인 lora 파라미터를 추가 한 후에 압축 -> 복원을 시켜서 전체 모델 구조에서의 출력값 형태와 계산을 유지하고, lora 파라미터만을 update 시키는 방식입니다. 

그렇기 때문에 LoRA를 적용하면 모델의 총 파라미터수는 증가하지만 학습 파라미터수는 많이 적어지기 때문에 기존의 전체 파라미터에 대한 gradient 연산 등이 줄어들기 때문에 전체 연산량은 오히려 감소하게 됩니다.

이는 torchsummaryX 같은 라이브러리들을 이용하여 모델 정보를 통해 확인할 수 있습니다. torchsummaryX의 경우에는 lora_dropout값을 0.0으로 설정해줘야 필요 연산량 값이 정상 표시 됩니다.

LoRA를 적용할 때는 3가지 종류의 layer 상태에 대해 고려를 해야합니다.

1. LoRA layer : LoRA가 적용된 layer. 기존 파라미터수의 차원이 압축된 LoRA 부분이 추가되고, 기존 linear, embedding layer의 파라미터는 freeze 됩니다.
2. freezed layer : LoRA가 적용되지 않은 layer는 기본적으로 전부 freeze 됩니다.
3. unfreezed layer : LoRA를 적용시키지 않으면서 freeze시키지 않고 싶은 layer입니다. convert_lora.py 내에서 keep_layer로 사용되었습니다. finetuning layer등을 고려해볼 수 있습니다. LoRA를 제외한 다른 layer는 전부 freeze하는 게 기본 메커니즘이고, 이 부분은 제가 실험을 위해 예외처리를 해주기 위해 추가한 항목입니다.

전체 모델에서 어떤 부분에 LoRA layer를 적용할지는 특별히 정해져있지는 않습니다. 제가 찾아본 바로는 attention 부분의 query, key, value에만 LoRA를 적용한 경우가 많았습니다.

Koelectra의 NSMC acc가 90.63(공식 레포)인데, Koelectra의 query, key, value에 실험을 위해 추가로 fine-tuning부분의 classifier부분에 lora를 적용했을 시 acc 0.896 정도의 값이 나왔었습니다. 현재 작업 환경문제로 학습을 끝까지 시키지 못하였지만 더 적은 파라미터수로 비슷한 성능을 내고, 경우에 따라서는 fully fine-tuning하는 경우보다 성능이 더 좋은 경우도 있기 때문에 다양한 실험이 필요할 것 같습니다. 

## Personal Insight

Foundation model이 하나의 주요 트랜드가 되면서 초거대언어 모델을 각각 상황에 맞게 새로 학습시키는 것은 현실적으로 너무 많은 자원을 필요로 한다. 또한 GPT나 T5 처럼 자연어의 모든 Task들을 Text to Text의 방법을 통해 해결 하려고 할 때, 추가적인 지시문이나 예문을 넣는 prompt tuning 방식도 사용할 수 있지만, 이는 그만큼 실제 입력 길이가 짧아진다는 것을 의미한다. prompt tuning는 물론 유용하지만 GPT2에서 이야기하는 zero shot learning은 성능에 한계가 있다. 이러한 점들을 고려할 때 모델의 일부 파라미터를 수정하여 필요한 부분을 효율적으로 학습하는 방식은 하나의 초거대모델을 만든 후 다양한 Task에 미세조정이 필요할 경우에 매우 유용할 수 있다. 때문에 기존에 연구되오던 이 분야의 중요성이 더욱 커졌다. 아이디어 자체는 LoRA는 facebook이 아니라 microsoft에서 나왔지만 Albert, Poly Encoder에서 이야기하는 저차원 압축과 상당히 비슷한 것 같다. 모델의 파라미터수가 커지면서 너무 빨리 overfit되는 문제들도 자주 보이게 되는데 그러한 부분에서도 일부 파라미터만을 업데이트하는 방식이 효과가 있을지에 대해서도 생각해볼 만한 문제인 것 같고, LoRA이후의 같은 영역에서 더 좋은 성능을 낸 방법들에 대해서도 기회가 된다면 확인을 해보고 싶다. 여전히 발전되고 있는 분야이고, 초거대언어 모델이 앞으로 더욱 major해질 것이라고 생각하기 때문에 이 분야도 매우 중요한 분야라고 생각한다. 
