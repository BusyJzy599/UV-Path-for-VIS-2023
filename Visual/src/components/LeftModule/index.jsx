/*
 * @Date: 2022-04-17 18:27:09
 * @LastEditors: JZY
 * @LastEditTime: 2023-02-05 16:23:45
 * @FilePath: /project/Visual/src/components/LeftModule/index.jsx
 */

import React, { Component } from "react";
import * as d3 from "d3";
import ReactECharts from "echarts-for-react";
import { Row, Col, Card, Typography, Select } from "antd";
const mapLevel = ["Postive", "0.25", "0.45", "0.60", "0.75", "Negative"];
const category_name = ["no cancer", "cancer", "high cancer"];
const re_category_name = ["high cancer", "cancer", "no cancer"];
const category_color = ["#40b373", "#d4b446", "#ee8826"]; // and more

const { Text, Title } = Typography;

const category_number = 3
const metric_index = [0, 1, 2]
const metric_index_r = [2, 1, 0]



export default class LeftModule extends Component {
  constructor(props) {
    super(props);
    this.state = {
      lineOption: {},
      sankeyOption: {},
      confuseMatrixOption: {},
    };
  }
  componentDidMount = async () => {
    var epochData = [];
    var cmData = [];
    var matrixData = this.props.confusion_Data;
    console.log(matrixData)
    var nai = [0, 0];
    var epoch = this.props.current_iteration;
    // Confusion matrix
    for (var i = 0; i < category_number; i++) {
      for (var j = 0; j < category_number; j++) {
        cmData.push(
          [i, j, matrixData[category_number - 1 - j][i]]
        )
      }
    }
    console.log(cmData)
    for (var i = 0; i < this.props.current_iteration; i++) {
      epochData.push([
        this.props.epoch_Data.labeled[i],
        this.props.epoch_Data.unlabeled[i],
        this.props.epoch_Data.noise_in_labeled[i],
        this.props.epoch_Data.infor_in_unlabled[i],
      ]);
      nai[0] += this.props.epoch_Data.noise_in_labeled[i];
      nai[1] += this.props.epoch_Data.infor_in_unlabled[i];
    }
    // sankey
    var nameData = [
      {
        name: "Dataset",
        itemStyle: {
          color: "#90b7b7",
          borderColor: "#709191",
        },
      },
      {
        name: "labled",
        itemStyle: {
          color: "#3980dc",
          borderColor: "#306ab5",
        },
      },
      {
        name: "Unlabled",
        itemStyle: {
          color: "#084c93",
          borderColor: "#09417c",
        },
      },
      {
        name: "Labled_",
        itemStyle: {
          color: "#3980dc",
          borderColor: "#306ab5",
        },
      },
      {
        name: "Unlabled_",
        itemStyle: {
          color: "#084c93",
          borderColor: "#09417c",
        },
      },
      {
        name: "Noisy",
        itemStyle: {
          color: "#ef9a9a",
          borderColor: "#ea8383",
        },
      },
      {
        name: "Clean",
        itemStyle: {
          color: "#91cc75",
          borderColor: "#7bae63",
        },
      },
      // {
      //   name: "Infor",
      //   itemStyle: {
      //     color: "#91cc75",
      //     borderColor: "#7bae63",
      //   },
      // },
    ];
    var linkData = [
      {
        source: nameData[0].name,
        target: nameData[1].name,
        value: epochData[0][0],
      },
      {
        source: nameData[0].name,
        target: nameData[2].name,
        value: epochData[0][1],
      },
      {
        source: nameData[1].name,
        target: nameData[3].name,
        value: epochData[0][0],
      },
      {
        source: nameData[2].name,
        target: nameData[4].name,
        value: epochData[epoch - 1][1],
      },
      {
        source: nameData[3].name,
        target: nameData[5].name,
        value: nai[0]
      },
      {
        source: nameData[3].name,
        target: nameData[6].name,
        value: epochData[epoch - 1][0]-nai[0]
      },
      {
        source: nameData[2].name,
        target: nameData[3].name,
        value: epochData[0][1] - epochData[epoch - 1][1],
      },
    ];

    // line
    var index = [];
    var acc = [];
    var auc = [];
    for (var i = 0; i < epoch; i++) {
      index.push(this.props.epoch_Data.epoch[i]);
      acc.push(this.props.epoch_Data.acc[i]);
      auc.push(this.props.epoch_Data.auc[i]);
    }

    this.setState({
      confuseMatrixOption: {
        title: {
          text: "Confusion Matrix",
        },
        grid: {
          left: 20,
          top: 45,
          right: 0,
          bottom: 10,
        },
        tooltip: {},
        xAxis: {
          type: "category",
          data: metric_index,
          position: "top",
          splitArea: {
            show: true,
          },
        },
        yAxis: {
          type: "category",
          data: metric_index_r,
          splitArea: {
            show: true,
          },
        },
        visualMap: {
          min: 0,
          max:1500,
          calculable: true,
          show: false,
          inRange: {
            color: ["#D9E9FF", "#0B69E3"], // 修改热力图的颜色 淡蓝色=>深蓝色的过度
          },
          orient: "horizontal",
        },
        series: [
          {
            name: "Confusion Matrix",
            type: "heatmap",
            data: cmData,
            label: {
              show: true,
            },
            emphasis: {
              itemStyle: {
                shadowBlur: 10,
                shadowColor: "rgba(0, 0, 0, 0.5)",
              },
            },
          },
        ],
      },
      sankeyOption: {
        // title: {
        //   text: "Sample Flow",
        // },
        tooltip: {
          trigger: "item",
          triggerOn: "mousemove",
        },

        animation: false,
        series: [
          {
            type: "sankey",
            left: 20,
            top: 20,
            nodeGap: 10,
            emphasis: {
              focus: "adjacency",
            },
            data: nameData,
            links: linkData,
            orient: "vertical",
            label: {
              position: "top",
            },
            lineStyle: {
              color: "source",
              curveness: 0.5,
            },
          },
        ],
      },
      lineOption: {
        title: {
          text: "ACC & AUC",
          top: -5,
        },
        legend: {
          top: 20,
          textStyle: {
            fontSize: 12,
          },
        },
        grid: {
          left: "15%",
          top: "20%",
          right: 0,
          bottom: "12%",
        },
        tooltip: {
          trigger: "axis",
          position: function (point, params, dom, rect, size) {
            return [point[0] - 100, "0%"]; //返回x、y（横向、纵向）两个点的位置
          },
          axisPointer: {
            type: "shadow",
          },
        },
        xAxis: {
          type: "category",
          data: index,
        },
        yAxis: {
          splitLine: {
            show: false,
          },
          type: "value",
          min: function (value) {
            return value.min - 0.05;
          },
        },
        series: [
          {
            name: "acc",
            type: "line",
            showSymbol: false,
            data: acc,
          },
          {
            name: "auc",
            type: "line",
            showSymbol: false,
            data: auc,
          },
        ],
      },
    });


    console.log(this.state)

  };

  render() {
    return (
      <Card bordered={false} hoverable={true}>
        <Row gutter={5}>
          <Col span={24}>
            <ReactECharts
              option={this.state.confuseMatrixOption}
              style={{ height: "23vh", width: "11vw" }}
            />
          </Col>
          <Col span={15}>
            <Title level={4} > Sample Flow</Title>
          </Col>
          <Col>
            Noise Ratio:&nbsp;
            <Select
              defaultValue="0.3"
              size="small"
              options={[
                {
                  value: '0.1',
                  disabled: true,
                  label: '0.1',
                },
                {
                  value: '0.3',
                  label: '0.3',
                },
                {
                  value: '0.5',
                  disabled: true,
                  label: '0.5',
                },
                {
                  value: '0.7',
                  disabled: true,
                  label: '0.7',
                },
              ]}
            />
          </Col>
          <Col span={24}>
            <ReactECharts
              option={this.state.sankeyOption}
              style={{ height: "36vh", width: "11vw" }}
            />
          </Col>
          <Col span={24}>
            <ReactECharts
              className="line-chart"
              style={{ height: "24vh", width: "11vw" }}
              option={this.state.lineOption}
            />
          </Col>
        </Row>
      </Card>
    );
  }
}
