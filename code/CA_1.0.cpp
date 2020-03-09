#include <iostream>
#include <time.h>
#include <stdlib.h>
using namespace std;

#define MAPSIZE 52
#define FALL_THRESHOLD 0.8

//the state of the cell
enum State { fall, stick };
enum Type {sand, water};

//Ԫ������
typedef struct cell{
	Type type;
	char waterContent;
	char angle;
}CELL;

int main(void) {
	
	//��ͼ��
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
	
	//ע����ͼ��Сд�ô���һȦ��Ϊ�˷����жϣ�һ��Ԫ����״̬Ҫ����Χ9��Ԫ����
	for (int k = 1; k < 10; k++) {
		for (int i = 0; i < MAPSIZE-2; i++) {
			for (int j = 1; j < MAPSIZE; j++) {
				if (CA[i][j].type == sand) {
					//�����ɳ
					if (CA[i - 1][j].type == water) {
						//���½Ƕȣ���ʱֻ�к��ٵļ������·�ʽ
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
						//��ȫ�Ӵ���ˮ����ˮ��100%
						CA[i][j].waterContent = 100;
					}
					//�ڲ��ɳ����ÿһ�����һ�㺬ˮ����10%
					else {
						CA[i][j].waterContent = CA[i - 1][j].waterContent - 10;
					}
					//�������
					if (CA[i][j].waterContent == 100) {
						//һ����������ϽǶȼӳ�
						fallPossibility = rand() / (double)RAND_MAX + (90.0 - CA[i][j].angle) / 180.0;
						if (fallPossibility > FALL_THRESHOLD) {
							CA[i][j].type = water;
						}
					}
				}
			}
		}
		//��ӡ���յ�ͼ��*��ɳ�ӣ�-��ˮ
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