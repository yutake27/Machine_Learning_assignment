# Machine_Learning_assignment

機械学習のライブラリを使わずに様々な機械学習手法の実装を行う。




## 1. ロジスティック回帰を用いた二値分類

以下の二つの最適化手法を実装し収束までの速度を比較する。

* steepset gradient method

* Newton based method

  

## 1-4. マルチクラスロジスティック回帰

ロジスティック回帰をマルチクラスに対応させ、以下の二つの最適化手法を実装し収束までの速度を比較する。

- steepset gradient method

- Newton based method

  

## 2. lasso

lassoで線形回帰を行う。最適化手法にはproximal gradient method(**PG**)を用いる。

そしてハイパーパラメータ lambda を変えたときのregularization pathを表示する。



## 3. SVM

SVMの実装を行い、

* negative dual Lagrange function
* hinge loss関数と正則化項のsum

とiterationの関係を示す。

最適化手法には単純化のためにprojected gradient methodを用いる。



## 5. 非線形SVMを用いた二値分類

ガウシアンカーネルを用いて非線形SVMを構築する。

データの数、 ハイパーパラメータalphaを変更してその結果を示す。



## 6. 行列補完問題

nuclear normを用いてnullを含む不完全な行列の補完を行う。

最適化手法にはproximal gradient methodを用いる。

元の行列と補完した行列をsurface plottingにより表示する。
