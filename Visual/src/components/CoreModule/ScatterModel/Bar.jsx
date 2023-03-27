/*
 * @Date: 2022-07-21 11:08:42
 * @LastEditors: JZY
 * @LastEditTime: 2022-12-17 13:28:31
 * @FilePath: /project/Visual/src/components/CoreModule/ScatterModel/Bar.jsx
 */
import React, { Component } from "react";
import ReactECharts from "echarts-for-react";
import { message } from "antd";

export default class Bar extends Component {
  constructor(props) {
    super(props);

    this.state = {
      index: [],
      O2u: [],
      Fine: [],
      Grades: [],
      option: {},
      choosePatches: props.choosePatches,
    };
  }
  componentDidMount = () => {
    this.props.onChildEvent(this);
    this.drawChart(0, -1);
  };

  changeDeletePatches = (p) => {
    this.setState({
      choosePatches: p,
    });
  };

  changeBarRange = (id) => {
    this.drawChart(id, id);
  };
  selectBar = {
    click: async (e) => {
      const newTags = this.state.choosePatches.filter(
        (tag) => tag !== e.dataIndex
      );
      if (newTags.length < 10) {
        await newTags.push(e.dataIndex);
        await this.setState({
          choosePatches: newTags,
        });
      } else {
        await message.error("The selected image has reached the limitation!");
      }
      this.props.changeChoosePatches(newTags);
    },
  };
  drawChart = (start, end) => {
    var index = [];
    var patchNum = [];
    var O2u = [];
    var Fine = [];

    var img_len = Object.keys(this.props.WSI_Data.img_id).length;
    for (var i = 0; i < img_len; i++) {
      index.push(this.props.WSI_Data["img_id"][i]);
      patchNum.push(this.props.WSI_Data["patch_num"][i]);

      O2u.push(this.props.WSI_Data["o2us"][i]);
      Fine.push(this.props.WSI_Data["fines"][i]);
    }
    // normalize
    function normalize(arr) {
      let max = 0;
      let min=2;
      let t=0.005;
      arr.forEach(v => {
        max=Math.max(v,max);
        min=Math.min(min,v);
      })
      for (let i = 0; i < arr.length; i++) {
        arr[i] = (arr[i]+t-min) / (max-min);
      }
      return arr
    }

    if (end == -1) end = index.length - 1;
    this.setState({
      option: {
        title: {
          text: "Noise Metric for Images:",
        },
        legend: {
          top: 10,
          textStyle: {
            fontSize: 10,
          },
        },
        tooltip: {
          trigger: "axis",
          axisPointer: {
            type: "shadow",
          },
        },
        grid: {
          top: 30,
          bottom: 50,
          left: 30,
          right: 0,
        },
        dataZoom: [
          {
            type: "slider",
            xAxisIndex: 0,
            left: 15,
            bottom: 0,
            startValue: start,
            endValue: end,
          },
          {
            type: "inside",
          },

        ],
        xAxis: {
          data: index,
          silent: false,
          splitLine: {
            show: false,
          },
          splitArea: {
            show: false,
          },
        },
        yAxis: [
          {
            position: "left",
            type: "value",
            min: 0,
            max: 1.05,
            splitArea: {
              show: false,
            },
          },
        ],
        series: [
          {
            name: "Mean Fitting Score",
            type: "bar",
            data: normalize(O2u),
            large: true,
          },
          {
            name: "Mean Alignment Score",
            type: "bar",
            data: normalize(Fine),
            large: true,
          },
        ],
      },
    });

  };

  render() {
    return (
      <>
        <ReactECharts
          style={{ height: "23vh" }}
          option={this.state.option}
          onEvents={this.selectBar}
        />
      </>
    );
  }
}
