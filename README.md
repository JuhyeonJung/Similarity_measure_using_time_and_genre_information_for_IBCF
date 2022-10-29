# Similarity_measure_using_time_and_genre_information_for_IBCF
Author : Juhyeon Jung, Kyoungok Kim(coressponding author)

### 연구 동기
 IBCF에 대한 유사도 계산에 아이템의 평가 시간 및 장르를 통합한다는 측면에서, 이전 연구는 다음과 같은 몇 가지 한계점를 가지고 있다.

- 유사도 지표에 시간 요소을 포함할 때 지수 시간 감쇠 함수가 주로 사용되었으며, 그 성능은 감쇠율이 다른 감쇠 함수와 비교되지 않았다.
- 유사도 지표에 평가 시간 및 아이템 장르를 모두 사용하는 연구를 찾기 어렵다.
- PCC와 같은 제한된 유사도을 바탕으로 유사도 지표에 시간 및 장르 요소를 도입함으로써 성능 개선을 검증하였다.

 이러한 문제를 해결하기 위해 본 연구에서 제안된 접근 방식은 다음과 같이 설계되었다.

- 서로 다른 감쇠율을 갖는 다른 시간 감쇠 함수는 시간 요소에 대한 영향을 제어하기 위해 사용되었다.
- JAC의 크기가 다른 lower bound는 장르 요소의 영향을 제어하기 위해 사용되었으며, 이는 장르 요소의 영향을 제어하기 위해 사용되었다.
- 단순히 시간과 장르 요소을 곱하는 것이 아니라 시간 요소를 파라미터로 이용한 상향된 장르 요소를 사용하여 유사도 지표에 아이템의 평가 시간과 장르를 모두 효과적으로 반영하였다.
- 다양한 전통적인 유사도 지표는 시간 및 장르 요소을 결합하여 기본 유사도 지표 따라 변화하는 예측 정확도 향상의 차이를 확인하는 기본 유사도로 사용되었다.


### 1. Time Decay Functions 정의
* Exponential 
$f(u,x,y)^{EXP}=e^{-\lambda \cdot \vert t_{u,x}-t_{u,y} \vert}, \lambda=\frac{1}{T_0}$  
* Sigmoid
$f(u,x,y)^{SIG}=\frac{2}{1+e^{\lambda \cdot\vert t_{u,x}-t_{u,y} \vert}}, \lambda=\frac{1}{T_0}$  
* Linear
$f(u,x,y)^{LIN}=(\vert t_{u,x}-t_{u,y} \vert+1)^{-\lambda}, \lambda=\frac{1}{T_0}$

![initial](https://user-images.githubusercontent.com/72389445/198817190-afd082a3-399e-4076-aace-68c8b5ecb92e.png)

아이템 평가 시점 차이에 따른 각 함수의 감쇠율

### 2. Genre Similarity
$sim(x,y)^{JAC}=1-\frac{NTF+NFT}{NTF+NFT+NTT}$  

𝑁𝑇𝑇 : number of dims in which both values are True  
𝑁𝑇𝐹 :  number of dims in which the first value is True, second is False  
𝑁𝐹𝑇 : number of dims in which the first value is False, second is True  

JAC의 upper bound를 0.99로 설정, 장르 요소의 영향을 제어하기 위해 장르 유사도에 대한 lower bound를 조절
$g(x,y)=(0.99-g_0)\cdot sim(x,y)^{JAC}+g_0$

### 3. Proposed Similarity

$sim(u,x,y)$를 정의하여 제안된 유사도 지표 정의 

* $sim(u,x,y)^{PCC} =\frac{(r_{u,x}-\bar{r_x})\cdot(r_{u,y}-\bar{r_y}) }{\sqrt{\sum_{v\in U_x\cap U_y}(r_{v,x}-\bar{r_x})^2}\sqrt{\sum_{v\in U_x\cap U_y}(r_{v,y}-\bar{r_y})^2}}$

* $sim(u,x,y)^{COS}=\frac{r_{u,x}\cdot r_{u,y} }{\sqrt{\sum_{v\in U_x\cap U_y}r_{v,x}^2}\sqrt{\sum_{v\in U_x\cap U_y}r_{v,y}^2}}$

* $sim(u,x,y)^{ACOS}=\frac{(r_{u,x}-\bar{r_u})\cdot(r_{u,y}-\bar{r_u}) }{\sqrt{\sum_{v\in U_x\cap U_y}(r_{v,x}-\bar{r_v})^2}\sqrt{\sum_{v\in U_x\cap U_y}(r_{v,y}-\bar{r_v})^2}}$

* $sim(u,x,y)^{MSD}=\frac{(1-(r_{u,x}-r_{u,y})^2) }{\vert U_x\cap U_y \vert}$

$sim(u,x,y)$를 정의하여 전통적인 유사도 지표 재정의 
*  $sim(x,y) = \sum_{u\in U_x\cap U_y} sim(u,x,y)$

***시간 요소를 결합한 유사도***
* $sim(x,y)~t= \sum_{u\in U_x\cap U_y}sim(u,x,y) \cdot f(u,x,y)$

***장르 요소를 결합한 유사도***  
* $sim(x,y)~g = \sum_{u\in U_x\cap U_y}sim(u,x,y) \cdot g(x,y)$

***시간 및 장르 요소를 동시에 결합한 유사도***   
시간과 장르 요소를 동시에 결합할때 단순 곱하는 형태 $g(x,y)\cdot f(u,x,y)$가 아닌 새로운 형태 $g(x,y)^{\frac{1}{f(u,x,y)+e}},e=0.5,1$ 제안
![initial](https://user-images.githubusercontent.com/72389445/198817905-fee909fb-a038-4d08-9dbf-a0035f75b912.png)  
시간 요소와 장르 요소를 단순 곱하였을 때 아이템 간 평가 시간 차이에 따른 유사도 변화
![initial](https://user-images.githubusercontent.com/72389445/198817937-d4e6f72d-a2ad-498b-ba45-d17d1f5dd8f9.png)  
시간 요소와 장르 요소를 새로운 형태로 결합했을 때 아이템 간 평가 시간 차이에 따른 유사도 변화 : $e=0.5$일 때  

단순히 시간 및 장르 요소를 곱하면 장르 유사도 값에 무관하게 0으로 수렴하는 경향, 반면에 장르 요인에 시간 요인을 제곱하는 방식은 장르 정보와 시간에 대한 영향력이 어떠한 상황에서도 선명하게 나타남  
- 최종 수식 :  
$sim(x,y)~p=\sum_{u\in U_x\cap U_y}sim(u,x,y)\cdot g(x,y)^{\frac{1}{f(u,x,y)+e}}$



