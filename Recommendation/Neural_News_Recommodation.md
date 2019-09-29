### Neural News Recommendation with Long and Short-term User Representations
**Key problem**: A key problem in news recommendation is learning accurate user representations to capture their interests.
**Core method:** News encoder and user encoder

* **News encoder**
  + Embedding layer, converts a news title from a word sequence into a vector sequence.
  + CNN learns contextual word representations.
  + Attention layer to select important words in news titles to learn informative news representations.
* **User encoder**
  + The user encoder module is used to learn the representations of users from the representations of their browsed news. Different news browsed by the same user may have different informativeness for representing this user.

![1568984302439](/home/qiujunyan/.config/Typora/typora-user-images/1568984302439.png)

* **Click Predictor**

  It is used to predict the probability of a user clicking a candidate news based on their hidden representations. 