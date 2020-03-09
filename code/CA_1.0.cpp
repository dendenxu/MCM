#include <iostream>
#include <time.h>
#include <stdlib.h>
using namespace std;

#define MAPSIZE 52
#define FALL_THRESHOLD 0.8

//the state of the cell
enum State { fall, stick };
enum Type {sand, water};

//元胞定义
typedef struct cell{
	Type type;
	char waterContent;
	char angle;
}CELL;

int main(void) {
	
	//地图，
	CELL CA[MAPSIZE][MAPSIZE];
	srand((unsigned)time(0));
	double fallPossibility;
	int i;

	for (i = 0; i < MAPSIZE; i++) {
		for (int j = 0; j < MAPSIZE; j++) {
			if (i == 0) {
				CA[i][j].type = water;
			} 
			else {
				CA[i][j].type = sand;
			}
			CA[i][j].waterContent = 0;
			CA[i][j].angle = 0;
		}
	}
	
	//注：地图大小写得大了一圈，为了方便判断（一个元胞的状态要看周围9个元胞）
	for (int k = 1; k < 10; k++) {
		for (int i = 0; i < MAPSIZE-2; i++) {
			for (int j = 1; j < MAPSIZE; j++) {
				if (CA[i][j].type == sand) {
					//最外层沙
					if (CA[i - 1][j].type == water) {
						//更新角度，暂时只有很少的几个更新方式
						if (CA[i - 1][j - 1].type == sand && CA[i][j + 1].type == water) {
							/*
								* - -
								* * -
								* * *
							*/
							if (CA[i + 1][j + 1].type == sand) {
								CA[i][j].angle = 45;
							}
							/*
								* - -
								* * -
								* * -
							*/
							else if (CA[i + 1][j + 1].type == water) {
								CA[i][j].angle = 30;
							}
						}
						else if (CA[i - 1][j + 1].type == sand && CA[i][j - 1].type == water) {
							if (CA[i + 1][j - 1].type == sand) {
								/*
									- - *
									- * *
									* * *
								*/
								CA[i][j].angle = 45;
							}
							else if (CA[i + 1][j - 1].type == water) {
								CA[i][j].angle = 30;
								/*
									- - *
									- * *
									- * *
								*/
							}
						}
						else {
							CA[i][j].angle = 90;
						}
						//完全接触到水，含水量100%
						CA[i][j].waterContent = 100;
					}
					//内层的沙粒：每一层比上一层含水量减10%
					else {
						CA[i][j].waterContent = CA[i - 1][j].waterContent - 10;
					}
					//掉落控制
					if (CA[i][j].waterContent == 100) {
						//一个随机数加上角度加成
						fallPossibility = rand() / (double)RAND_MAX + (90.0 - CA[i][j].angle) / 180.0;
						if (fallPossibility > FALL_THRESHOLD) {
							CA[i][j].type = water;
						}
					}
				}
			}
		}
		//打印最终地图：*是沙子，-是水
		for (int i = 0; i < 50; i++) {
			for (int j = 1; j < 51; j++) {
				if (CA[i][j].type == sand) {
					cout << "* ";
				}
				else {
					cout << "- ";
				}
			}
			cout << "" << endl;
		}
		cout << "" << endl;
	}

	return 0;
}