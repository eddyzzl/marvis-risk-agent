(function installPetCatalog(global) {
  "use strict";

  if (global.MarvisPetCatalog) return;

  const schemaVersion = 1;
  const defaultId = "auditbot";
  const noneId = "none";
  const storageKeys = Object.freeze({
    preference: "marvis_pet",
    explicitNone: "marvis_pet_none_explicit",
    position: "marvis_pet_position",
  });
  const aliases = Object.freeze({
    danhuang: "naitang",
    buou: "xiaojiu",
    "ragdoll-cat": "xiaojiu",
  });
  const order = Object.freeze([
    noneId,
    "naitang",
    "xiaojiu",
    defaultId,
    "auditbot-pro",
    "auditbot-poly",
    "auditbot-ink",
    "auditbot-clay",
    "auditbot-comic",
    "auditbot-pixel",
  ]);
  const definitions = Object.freeze({
    none: Object.freeze({
      name: "不显示",
      label: "不显示宠物",
      kind: "none",
      asset: null,
    }),
    naitang: Object.freeze({
      name: "蛋黄",
      label: "奶油色长毛蓝眼猫，黑色领结",
      kind: "spritesheet",
      asset: "static/pets/naitang/spritesheet.webp",
    }),
    xiaojiu: Object.freeze({
      name: "小九",
      label: "贪吃、呆萌、胆小的小猫",
      kind: "spritesheet",
      asset: "static/pets/xiaojiu/spritesheet.webp?v=c078ec6f",
    }),
    auditbot: Object.freeze({
      name: "MARVIS",
      label: "3D 玩具审计机器人，青色护目镜眼睛和铜色耳机",
      kind: "spritesheet",
      asset: "static/pets/auditbot/spritesheet.webp",
    }),
    "auditbot-pro": Object.freeze({
      name: "MARVIS Pro",
      label: "专业风格 3D 审计机器人",
      kind: "spritesheet",
      asset: "static/pets/auditbot-pro/spritesheet.webp",
    }),
    "auditbot-poly": Object.freeze({
      name: "MARVIS Poly",
      label: "低多边形硬表面审计机器人",
      kind: "spritesheet",
      asset: "static/pets/auditbot-poly/spritesheet.webp",
    }),
    "auditbot-ink": Object.freeze({
      name: "MARVIS Ink",
      label: "技术线稿风格审计机器人",
      kind: "spritesheet",
      asset: "static/pets/auditbot-ink/spritesheet.webp",
    }),
    "auditbot-clay": Object.freeze({
      name: "MARVIS Clay",
      label: "黏土与乙烯基质感审计机器人",
      kind: "spritesheet",
      asset: "static/pets/auditbot-clay/spritesheet.webp",
    }),
    "auditbot-comic": Object.freeze({
      name: "MARVIS Comic",
      label: "漫画描边风格审计机器人",
      kind: "spritesheet",
      asset: "static/pets/auditbot-comic/spritesheet.webp",
    }),
    "auditbot-pixel": Object.freeze({
      name: "MARVIS Pixel",
      label: "像素风审计机器人",
      kind: "spritesheet",
      asset: "static/pets/auditbot-pixel/spritesheet.webp",
    }),
  });

  function normalizePreference(value) {
    if (value === noneId) return noneId;
    const normalized = aliases[value] || value;
    return definitions[normalized]?.asset ? normalized : defaultId;
  }

  function resolveStoredPreference(value, explicitNone) {
    if (!value || (value === noneId && !explicitNone)) return defaultId;
    return normalizePreference(value);
  }

  function populateSelect(select, documentRef = global.document) {
    if (!select || !documentRef) return;
    const options = order.map((petId) => {
      const option = documentRef.createElement("option");
      option.value = petId;
      option.textContent = definitions[petId].name;
      return option;
    });
    select.replaceChildren(...options);
  }

  global.MarvisPetCatalog = Object.freeze({
    schemaVersion,
    defaultId,
    noneId,
    storageKeys,
    aliases,
    order,
    definitions,
    normalizePreference,
    resolveStoredPreference,
    populateSelect,
  });
})(globalThis);
