/*
 * @Date: 2022-07-21 11:22:01
 * @LastEditors: JZY
 * @LastEditTime: 2023-02-05 16:06:11
 * @FilePath: /project/Visual/src/components/CoreModule/SankeyModel/index.jsx
 */
import React, { Component } from "react";
import * as d3 from "d3";
import ReactECharts from "echarts-for-react";
import { Typography, Slider, Col, Row } from "antd";
const { Title, Text } = Typography;

const category_name = ["no cancer", "cancer"];

export default class SankeyModel extends Component {
  constructor(props) {
    super(props);
    this.state = {
      sankeyOption: {},
      confuseMatrixOption: {},
      epochData: [],
      matrixData: [],
      marks: {},
    };
  }
  componentDidMount = async () => {
    var epochData = [];
    var matrixData = [];
    var marks = {};

  
    for(var i=0;i< this.props.current_iteration;i++){
      epochData.push([
        this.props.epoch_Data.labeled[i],
        this.props.epoch_Data.unlabeled[i],
        this.props.epoch_Data.noise_in_labeled[i],
        this.props.epoch_Data.infor_in_unlabled[i],
      ])
      matrixData.push([
        this.props.epoch_Data.TP[i],
        this.props.epoch_Data.FP[i],
        this.props.epoch_Data.TN[i],
        this.props.epoch_Data.FN[i],

      ])
    }
    for (var i = 1; i <= this.props.current_iteration; i++) {
      marks[i] = i + "th";
    }
    this.setState({
      epochData: epochData,
      matrixData: matrixData,
      marks: marks,
    });
    setTimeout(() => {
      this.drawMatric(-1);
      this.drawSankey(-1);
    }, 0);
  };

  drawSankey = (epoch) => {
    if (epoch == -1) epoch = this.state.epochData.length-1;
    var nameData = [];
    var linkData = [];
    // nameData.push({
    //   name: "Labeled_",
    //   itemStyle: {
    //     color: "#f18bbf",
    //     borderColor: "#f18bbf",
    //   },
    // });
    // nameData.push({
    //   name: "Unlabeled_",
    //   itemStyle: {
    //     color: "#0078D7",
    //     borderColor: "#0078D7",
    //   },
    // });
    // nameData.push({
    //   name: "Noise_" ,
    //   itemStyle: {
    //     color: "#3891A7",
    //     borderColor: "#3891A7",
    //   },
    // });
    // nameData.push({
    //   name: "Info_" ,
    //   itemStyle: {
    //     color: "#C0BEAF",
    //     borderColor: "#C0BEAF",
    //   },
    // });
    // nameData.push({
    //   name: "Labeled_1",
    //   itemStyle: {
    //     color: "#f18bbf",
    //     borderColor: "#f18bbf",
    //   },
    // });
    // nameData.push({
    //   name: "Unlabeled_1",
    //   itemStyle: {
    //     color: "#0078D7",
    //     borderColor: "#0078D7",
    //   },
    // });
    // nameData.push({
    //   name: "Noise_1" ,
    //   itemStyle: {
    //     color: "#3891A7",
    //     borderColor: "#3891A7",
    //   },
    // });
    // nameData.push({
    //   name: "Info_1" ,
    //   itemStyle: {
    //     color: "#C0BEAF",
    //     borderColor: "#C0BEAF",
    //   },
    // });
    // var i=0
    // linkData.push(
    //   {
    //     source: nameData[4 * i].name,
    //     target: nameData[4 * (i + 1)].name,
    //     value:
    //       this.state.epochData[i][0] -
    //       (this.state.epochData[i + epoch][2] - this.state.epochData[i][2]),
    //   },
    //   {
    //     source: nameData[4 * i + 1].name,
    //     target: nameData[4 * (i + 1) + 1].name,
    //     value: this.state.epochData[i + epoch][1],
    //   },
    //   {
    //     source: nameData[4 * i + 2].name,
    //     target: nameData[4 * (i + 1) + 2].name,
    //     value: this.state.epochData[i][2],
    //   },
    //   {
    //     source: nameData[4 * i + 3].name,
    //     target: nameData[4 * (i + 1) + 3].name,
    //     value: this.state.epochData[i][3],
    //   },

    //   {
    //     source: nameData[4 * i + 1].name,
    //     target: nameData[4 * (i + 1)].name,
    //     value:
    //       this.state.epochData[i][1] -
    //       this.state.epochData[i + epoch][1] -
    //       (this.state.epochData[i + epoch][3] - this.state.epochData[i][3]),
    //   },
    //   {
    //     source: nameData[4 * i + 1].name,
    //     target: nameData[4 * (i + 1) + 3].name,
    //     value: this.state.epochData[i + epoch][3] - this.state.epochData[i][3],
    //   },

    //   {
    //     source: nameData[4 * i].name,
    //     target: nameData[4 * (i + 1) + 2].name,
    //     value: this.state.epochData[i + epoch][2] - this.state.epochData[i][2],
    //   }
    // );

    for (var i = 0; i <= epoch; i++) {
      nameData.push({
        name: "Labeled_" + i,
        itemStyle: {
          color: "#f18bbf",
          borderColor: "#f18bbf",
        },
      });
      nameData.push({
        name: "Unlabeled_" + i,
        itemStyle: {
          color: "#0078D7",
          borderColor: "#0078D7",
        },
      });
      nameData.push({
        name: "Noise_" + i,
        itemStyle: {
          color: "#3891A7",
          borderColor: "#3891A7",
        },
      });
      nameData.push({
        name: "Info_" + i,
        itemStyle: {
          color: "#C0BEAF",
          borderColor: "#C0BEAF",
        },
      });
    }

    for (var i = 0; i < epoch; i++) {
      linkData.push(
        {
          source: nameData[4 * i].name,
          target: nameData[4 * (i + 1)].name,
          value:
            this.state.epochData[i][0] -
            (this.state.epochData[i + 1][2] - this.state.epochData[i][2]),
        },
        {
          source: nameData[4 * i + 1].name,
          target: nameData[4 * (i + 1) + 1].name,
          value: this.state.epochData[i + 1][1],
        },
        {
          source: nameData[4 * i + 2].name,
          target: nameData[4 * (i + 1) + 2].name,
          value: this.state.epochData[i][2],
        },
        {
          source: nameData[4 * i + 3].name,
          target: nameData[4 * (i + 1) + 3].name,
          value: this.state.epochData[i][3],
        },

        {
          source: nameData[4 * i + 1].name,
          target: nameData[4 * (i + 1)].name,
          value:
            this.state.epochData[i][1] -
            this.state.epochData[i + 1][1] -
            (this.state.epochData[i + 1][3] - this.state.epochData[i][3]),
        },
        {
          source: nameData[4 * i + 1].name,
          target: nameData[4 * (i + 1) + 3].name,
          value: this.state.epochData[i + 1][3] - this.state.epochData[i][3],
        },

        {
          source: nameData[4 * i].name,
          target: nameData[4 * (i + 1) + 2].name,
          value: this.state.epochData[i + 1][2] - this.state.epochData[i][2],
        }
      );
    }
    console.log("linkData111",linkData)
    this.setState({
      sankeyOption: {
        tooltip: {
          trigger: "item",
          triggerOn: "mousemove",
        },
        animation: false,

        series: [
          {
            type: "sankey",
            right:10,
            nodeGap: 10,
            left: "1%",
            emphasis: {
              focus: "adjacency",
            },
            data: nameData,
            links: linkData,
            label: {
              show: false,
              position: "topLeft",
            },
            lineStyle: {
              color: "source",
              curveness: 0.5,
            },
          },
        ],
      },
    });
  };
  drawMatric = (epoch) => {
    if (epoch == -1) epoch = this.state.matrixData.length - 1;
    this.setState({
      confuseMatrixOption: {
        grid: {
          // height: '40%',
          left: "32%",
          top: "25%",
          bottom: 10,
        },
        tooltip: {},
        xAxis: {
          type: "category",
          data: category_name,
          position: "top",
          splitArea: {
            show: true,
          },
        },
        yAxis: {
          type: "category",
          data: category_name,
          splitArea: {
            show: true,
          },
        },
        visualMap: {
          min: 0,
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
            data: [
              [0, 0, this.state.matrixData[epoch][0].toFixed(1)],
              [0, 1, this.state.matrixData[epoch][1].toFixed(1)],
              [1, 1, this.state.matrixData[epoch][2].toFixed(1)],
              [1, 0, this.state.matrixData[epoch][3].toFixed(1)],
            ],
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
    });
  };
  seletctMatrixEpoch = (value) => {
    this.drawMatric(value-1);
  };
  seletctSankeyEpoch = async (value) => {
    this.drawSankey(value-1);
  };

  render() {
    return (
      <>
        <Row gutter={5}>
          <Col span={19}>
            {/* <Title level={5}>SanKey for Epoches:</Title> */}
            <Slider
              className="slider-sankey"
              marks={this.state.marks}
              min={1}
              max={this.props.current_iteration}
              defaultValue={this.props.current_iteration}
              step={null}
              onChange={this.seletctSankeyEpoch}
            />
            <ReactECharts
              option={this.state.sankeyOption}
              style={{ height: "16vh", width: "58vw" }}
            />
          </Col>
          <Col span={5} id="ConfusionChart">
            {this.props.mapValid ? null : (
              <>
                <ReactECharts
                  option={this.state.confuseMatrixOption}
                  style={{ height: "18vh", width: "14vw" }}
                />
                <Row>
                  <Col span={1} />
                  <Col span={6}>
                    <Text type="secondary">Epoch:</Text>
                  </Col>
                  <Col span={17}>
                    <Slider
                      included={false}
                      defaultValue={this.props.current_iteration}
                      min={1}
                      max={this.props.current_iteration }
                      onChange={this.seletctMatrixEpoch}
                    />
                  </Col>
                </Row>
              </>
            )}
          </Col>
        </Row>
      </>
    );
  }
}
